#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General upper-body policy action generator.

This module keeps policy execution robot-agnostic. Robot-specific ROS topics,
message types, joint order, and virtual gripper presets live in a robot JSON
file. Model path, observation layout, action layout, reset pose, and completion
conditions live in a policy JSON file.
"""

import argparse
import json
import re
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from robot_adapter_class import (
    CAMERA_COMPONENTS,
    COMPONENT_TO_KEY,
    GRIPPER_TO_HAND,
    SUPPORTED_ADAPTERS,
    RobotComponent,
    component_from_name as _component_from_name,
    component_list as _component_list,
    component_names as _component_names,
    create_robot_adapter,
    has_component as _has,
    joint_count as _joint_count,
    validate_robot_schema as _validate_robot_schema,
)

try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
except ImportError as exc:  # Allows --dry-run in non-ROS environments.
    rclpy = None
    MultiThreadedExecutor = None
    QoSProfile = None
    ReliabilityPolicy = None
    HistoryPolicy = None
    DurabilityPolicy = None
    _ROS_IMPORT_ERROR = exc

    class Node:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "ROS2 Python packages are required to instantiate "
                "GeneralPolicyActionGenerator. Use --dry-run for config checks."
            ) from _ROS_IMPORT_ERROR


class CompleteCondition(Enum):
    """Policy execution completion conditions."""

    TIME_BASED = 0
    JOINT_REACHED = 6
    CONDITION_1 = 1
    CONDITION_2 = 2
    CONDITION_3 = 3
    CONDITION_4 = 4
    CONDITION_5 = 5


@dataclass
class JointReachedCondition:
    """Position threshold plus settle-time completion condition."""

    target: Dict[str, Any]
    threshold: float = 0.05
    settle_time: float = 0.3


@dataclass
class ImageSpec:
    component: RobotComponent
    obs_key: str
    target_size: Optional[Tuple[int, int]] = None


@dataclass
class LoadedConfigs:
    robot: Dict[str, Any]
    policy: Dict[str, Any]
    robot_path: Path
    policy_path: Path
    state_order: List[RobotComponent]
    action_order: List[RobotComponent]
    image_specs: List[ImageSpec]


def _load_json(path: Union[str, Path]) -> Dict[str, Any]:
    json_path = Path(path).expanduser().resolve()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{json_path} must contain a JSON object")
    return data


def _resolve_path(path_value: str, config_path: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    config_relative = (config_path.parent / path).resolve()
    if config_relative.exists():
        return config_relative
    return (Path.cwd() / path).resolve()


def _normalize_image_specs(raw_images: Any) -> List[ImageSpec]:
    specs: List[ImageSpec] = []
    if isinstance(raw_images, dict):
        # {"camera_head": "CAMERA_HEAD", "camera_wrist": "CAMERA_HAND_RIGHT"}
        for obs_key, component_name in raw_images.items():
            specs.append(ImageSpec(_component_from_name(component_name), str(obs_key)))
        return specs

    if isinstance(raw_images, list):
        for item in raw_images:
            if not isinstance(item, dict):
                raise ValueError("observation.images list entries must be objects")
            component = _component_from_name(item["component"])
            obs_key = str(item.get("obs_key", item.get("key", COMPONENT_TO_KEY[component])))
            target_size = item.get("target_size")
            size_tuple = tuple(target_size) if target_size is not None else None
            specs.append(ImageSpec(component, obs_key, size_tuple))  # type: ignore[arg-type]
        return specs

    raise ValueError("observation.images must be a list or object")


def _validate_no_hand_gripper_conflict(action_order: List[RobotComponent]) -> None:
    action_set = set(action_order)
    for gripper, hand in GRIPPER_TO_HAND.items():
        if gripper in action_set and hand in action_set:
            raise ValueError(
                f"action.order cannot include both {hand.name} and {gripper.name}; "
                "use either direct hand joints or virtual gripper control for that side"
            )


def _validate_component_value_length(
    robot_config: Dict[str, Any],
    component: RobotComponent,
    value: Any,
    context: str,
) -> None:
    if value is None:
        return
    size = _joint_count(robot_config, component)
    if component in (RobotComponent.LEFT_GRIPPER, RobotComponent.RIGHT_GRIPPER):
        arr = np.asarray(value, dtype=object)
        if arr.ndim == 0:
            return
        if len(arr) == 1:
            return
        raise ValueError(f"{context}.{component.name} must be a scalar or length-1 value")

    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        raise ValueError(f"{context}.{component.name} must be length {size}, got scalar")
    if len(arr) != size:
        raise ValueError(f"{context}.{component.name} must be length {size}, got {len(arr)}")


def load_and_validate_configs(
    robot_config_path: Union[str, Path],
    policy_config_path: Union[str, Path],
) -> LoadedConfigs:
    """Load and validate robot and policy JSON files."""

    robot_path = Path(robot_config_path).expanduser().resolve()
    policy_path = Path(policy_config_path).expanduser().resolve()
    robot = _load_json(robot_path)
    policy = _load_json(policy_path)

    _validate_robot_schema(robot)
    if robot["adapter"] not in SUPPORTED_ADAPTERS:
        raise ValueError(f"Unsupported robot adapter: {robot['adapter']}")

    for key in ["model_path", "observation", "action"]:
        if key not in policy:
            raise ValueError(f"policy config missing required key: {key}")
    if "state_order" not in policy["observation"]:
        raise ValueError("policy config missing observation.state_order")
    if "images" not in policy["observation"]:
        raise ValueError("policy config missing observation.images")
    if "order" not in policy["action"]:
        raise ValueError("policy config missing action.order")

    state_order = _component_list(policy["observation"]["state_order"])
    action_order = _component_list(policy["action"]["order"])
    image_specs = _normalize_image_specs(policy["observation"]["images"])

    for component in action_order:
        if _has(CAMERA_COMPONENTS, component):
            raise ValueError(f"Camera component cannot be in action.order: {component.name}")
    for component in state_order:
        if _has(CAMERA_COMPONENTS, component):
            raise ValueError(
                f"Camera component cannot be in observation.state_order: {component.name}; "
                "put cameras under observation.images"
            )
    _validate_no_hand_gripper_conflict(action_order)

    for component in state_order + action_order:
        _joint_count(robot, component)

    camera_topics = robot.get("topics", {}).get("cameras", {})
    for spec in image_specs:
        if spec.component not in (
            RobotComponent.CAMERA_HEAD,
            RobotComponent.CAMERA_HAND_LEFT,
            RobotComponent.CAMERA_HAND_RIGHT,
        ):
            raise ValueError(f"observation image component is not a camera: {spec.component.name}")
        if spec.component.name not in camera_topics:
            raise ValueError(f"robot config missing camera topic for {spec.component.name}")

    for gripper in (RobotComponent.LEFT_GRIPPER, RobotComponent.RIGHT_GRIPPER):
        if gripper in state_order or gripper in action_order:
            if gripper.name not in robot.get("grippers", {}):
                raise ValueError(f"robot config missing grippers.{gripper.name}")

    initial_pose = policy.get("initial_pose") or {}
    if not isinstance(initial_pose, dict):
        raise ValueError("policy initial_pose must be an object")
    for key, value in initial_pose.items():
        component = _component_from_name(key)
        if _has(CAMERA_COMPONENTS, component):
            raise ValueError(f"initial_pose cannot include camera component: {component.name}")
        _validate_component_value_length(robot, component, value, "initial_pose")

    for idx, condition in enumerate(policy.get("complete_conditions", [])):
        if condition.get("type") != "JOINT_REACHED":
            continue
        target = condition.get("target", {})
        if not isinstance(target, dict):
            raise ValueError(f"complete_conditions[{idx}].target must be an object")
        for key, value in target.items():
            component = _component_from_name(key)
            _validate_component_value_length(
                robot, component, value, f"complete_conditions[{idx}].target"
            )

    return LoadedConfigs(
        robot=robot,
        policy=policy,
        robot_path=robot_path,
        policy_path=policy_path,
        state_order=state_order,
        action_order=action_order,
        image_specs=image_specs,
    )


def describe_mapping(configs: LoadedConfigs) -> str:
    """Return a dry-run summary of vector mapping and robot topics."""

    lines = []
    robot = configs.robot
    policy = configs.policy
    lines.append(f"Robot: {robot['robot_name']} ({robot['adapter']})")
    lines.append(f"Robot config: {configs.robot_path}")
    lines.append(f"Policy config: {configs.policy_path}")
    lines.append(f"Model path: {policy['model_path']}")
    lines.append("")
    lines.append("Observation state vector:")
    idx = 0
    for component in configs.state_order:
        size = _joint_count(robot, component)
        lines.append(f"  {idx:02d}:{idx + size:02d}  {component.name} ({size})")
        idx += size
    lines.append(f"  total: {idx}")
    lines.append("")
    lines.append("Observation images:")
    for spec in configs.image_specs:
        topic = robot["topics"]["cameras"][spec.component.name]["topic"]
        todo = robot["topics"]["cameras"][spec.component.name].get("todo")
        suffix = f"  TODO: {todo}" if todo else ""
        lines.append(f"  {spec.obs_key} <- {spec.component.name} ({topic}){suffix}")
    lines.append("")
    lines.append("Action vector:")
    idx = 0
    for component in configs.action_order:
        size = _joint_count(robot, component)
        lines.append(f"  {idx:02d}:{idx + size:02d}  {component.name} ({size})")
        idx += size
    lines.append(f"  total: {idx}")
    return "\n".join(lines)


class GeneralPolicyActionGenerator(Node):
    """
    Robot-agnostic upper-body policy executor.

    The public API mirrors policy_action_generator.py:
    start, asyncStart, stop, getTaskStatus, getProcessDone,
    setCompleteCondition, and setCompleteConditions.
    """

    _global_execution_lock = threading.Lock()
    _global_executing_instance = None

    def __init__(
        self,
        robot_config_path: Union[str, Path],
        policy_config_path: Union[str, Path],
        node_name_suffix: Optional[str] = None,
    ):
        if rclpy is None:
            raise RuntimeError("rclpy is required for runtime execution") from _ROS_IMPORT_ERROR

        base = "general_policy_action_generator"
        if node_name_suffix:
            safe = re.sub(r"[^a-zA-Z0-9_]", "_", str(node_name_suffix))[:32]
            node_name = f"{base}_{safe}"
        else:
            node_name = base
        super().__init__(node_name)

        self.configs = load_and_validate_configs(robot_config_path, policy_config_path)
        self.robot_config = self.configs.robot
        self.policy_config = self.configs.policy
        self.adapter_name = self.robot_config["adapter"]
        self.state_order = self.configs.state_order
        self.action_order = self.configs.action_order
        self.image_specs = self.configs.image_specs

        runtime_cfg = self.policy_config.get("runtime", {})
        control_cfg = self.robot_config.get("control", {})
        self.publish_period = 1.0 / float(runtime_cfg.get("publish_hz", control_cfg.get("publish_hz", 30.0)))
        self.warming_up_time = float(runtime_cfg.get("warming_up_time", 3.0))
        self._min_run_time_before_complete = float(runtime_cfg.get("min_run_time_before_complete", 5.0))
        self._reset_steps = int(runtime_cfg.get("reset_steps", control_cfg.get("reset_steps", 100)))
        self._reset_duration = float(runtime_cfg.get("reset_duration", control_cfg.get("reset_duration", 2.0)))
        self._state_wait_timeout = float(runtime_cfg.get("state_wait_timeout", 5.0))

        self._load_common_message_types()
        self.robot_io = create_robot_adapter(self)
        self.robot_io.load_message_types()

        from cv_bridge import CvBridge

        self.bridge = CvBridge()

        # Latest state caches.
        self.joint_state: Dict[RobotComponent, Optional[np.ndarray]] = {
            RobotComponent.HEAD: None,
            RobotComponent.LEFT_ARM: None,
            RobotComponent.RIGHT_ARM: None,
            RobotComponent.LEFT_HAND: None,
            RobotComponent.RIGHT_HAND: None,
        }
        self.gripper_state: Dict[RobotComponent, float] = {
            RobotComponent.LEFT_GRIPPER: 0.0,
            RobotComponent.RIGHT_GRIPPER: 0.0,
        }
        self.images: Dict[RobotComponent, Optional[np.ndarray]] = {
            RobotComponent.CAMERA_HEAD: None,
            RobotComponent.CAMERA_HAND_LEFT: None,
            RobotComponent.CAMERA_HAND_RIGHT: None,
        }

        # ROS entities.
        self.publishers: Dict[str, Any] = {}
        self.subscriptions = []
        self.robot_io.setup_ros_interfaces()
        self._setup_camera_interfaces()

        # Load policy after config and ROS setup validation.
        from action_policy import PolicyAgent

        model_path = _resolve_path(str(self.policy_config["model_path"]), self.configs.policy_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Policy model not found: {model_path}")
        self.get_logger().info(f"Loading policy from: {model_path}")
        self.action_policy = PolicyAgent(str(model_path))

        self.action_queue = deque(maxlen=int(runtime_cfg.get("action_queue_size", 100)))
        self.action_queue_lock = threading.Lock()
        self.queue_updated = threading.Event()
        self.last_action = None
        self.actions_consumed_since_last_inference = 0
        self.last_n_consumed = 0
        self.inference_running = False
        self.publish_running = False

        self._execution_running = False
        self._inference_thread = None
        self._publish_thread = None
        self._stop_requested = False
        self._current_task_status = ""
        self._process_done = True
        self._state_lock = threading.Lock()
        self._start_time = 0.0
        self._inference_start_time: Optional[float] = None
        self._complete_check_eligible_time: Optional[float] = None
        self._inference_count = 0
        self._condition_progress = 0.0
        self._joint_reached_settle_starts: Dict[int, Optional[float]] = {}
        self._active_complete_conditions = self._parse_complete_conditions_from_policy()

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

        self.get_logger().info("General policy action generator initialized")
        self.get_logger().info(f"  Robot: {self.robot_config['robot_name']} ({self.adapter_name})")
        self.get_logger().info(f"  Observation: {_component_names(self.state_order)}")
        self.get_logger().info(f"  Action: {_component_names(self.action_order)}")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_common_message_types(self):
        from sensor_msgs.msg import Image, JointState
        from std_msgs.msg import Header, Int32

        self.ImageMsg = Image
        self.JointStateMsg = JointState
        self.HeaderMsg = Header
        self.Int32Msg = Int32

    def _setup_camera_interfaces(self):
        cameras = self.robot_config["topics"]["cameras"]
        for spec in self.image_specs:
            topic = cameras[spec.component.name]["topic"]
            self.subscriptions.append(
                self.create_subscription(
                    self.ImageMsg,
                    topic,
                    self._make_camera_callback(spec.component, spec.target_size),
                    10,
                )
            )
            self.get_logger().info(f"Subscribed image {spec.component.name}: {topic}")

    def make_qos(self, cfg: Optional[Dict[str, Any]]):
        if not cfg:
            return 10
        reliability = str(cfg.get("reliability", "RELIABLE")).upper()
        durability = str(cfg.get("durability", "VOLATILE")).upper()
        history = str(cfg.get("history", "KEEP_LAST")).upper()
        depth = int(cfg.get("depth", 10))
        return QoSProfile(
            reliability=getattr(ReliabilityPolicy, reliability),
            durability=getattr(DurabilityPolicy, durability),
            history=getattr(HistoryPolicy, history),
            depth=depth,
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _make_camera_callback(self, component: RobotComponent, target_size: Optional[Tuple[int, int]]):
        def _callback(msg):
            try:
                image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                if target_size is not None:
                    width, height = target_size
                    if image.shape[1] != width or image.shape[0] != height:
                        image = cv2.resize(image, (width, height))
                self.images[component] = image
            except Exception as exc:
                self.get_logger().error(f"Error converting {component.name}: {exc}")

        return _callback

    def make_scalar_gripper_callback(self, component: RobotComponent):
        def _callback(msg):
            self.gripper_state[component] = float(msg.data)

        return _callback

    def make_hand_joint_state_callback(self, component: RobotComponent):
        def _callback(msg):
            self.joint_state[component] = np.array(list(msg.position), dtype=np.float64)

        return _callback

    # ------------------------------------------------------------------
    # Observation and action mapping
    # ------------------------------------------------------------------

    def gripper_config(self, component: RobotComponent) -> Dict[str, Any]:
        return self.robot_config.get("grippers", {}).get(component.name, {})

    def _component_state(self, component: RobotComponent) -> Optional[np.ndarray]:
        if component in (RobotComponent.LEFT_GRIPPER, RobotComponent.RIGHT_GRIPPER):
            return np.array([self._virtual_gripper_value(component)], dtype=np.float64)
        if component in self.joint_state:
            state = self.joint_state[component]
            return None if state is None else np.array(state, dtype=np.float64)
        raise ValueError(f"Component has no state vector: {component.name}")

    def _virtual_gripper_value(self, component: RobotComponent) -> float:
        cfg = self.gripper_config(component)
        source = cfg.get("state_source", "last_command")
        if source == "scalar_topic":
            return float(self.gripper_state.get(component, 0.0))
        if source == "hand_average":
            hand = GRIPPER_TO_HAND[component]
            hand_state = self.joint_state.get(hand)
            if hand_state is None:
                return float(self.gripper_state.get(component, 0.0))
            open_preset = np.asarray(cfg.get("open_preset", np.zeros_like(hand_state)), dtype=np.float64)
            close_preset = np.asarray(cfg.get("close_preset", np.ones_like(hand_state)), dtype=np.float64)
            if len(open_preset) != len(hand_state) or len(close_preset) != len(hand_state):
                return float(self.gripper_state.get(component, 0.0))
            d_open = float(np.linalg.norm(hand_state - open_preset))
            d_close = float(np.linalg.norm(hand_state - close_preset))
            denom = d_open + d_close
            if denom <= 1e-9:
                return 0.0
            return float(np.clip(d_open / denom, 0.0, 1.0))
        return float(self.gripper_state.get(component, 0.0))

    def get_current_state(self) -> np.ndarray:
        parts = []
        for component in self.state_order:
            state = self._component_state(component)
            if state is None:
                parts.append(np.zeros(_joint_count(self.robot_config, component), dtype=np.float64))
            else:
                parts.append(state)
        return np.concatenate(parts) if parts else np.array([], dtype=np.float64)

    def get_obs(self) -> Optional[Dict[str, Any]]:
        obs = {"images": {}, "arm_gripper_joints": None}
        for spec in self.image_specs:
            image = self.images.get(spec.component)
            if image is None:
                return None
            obs["images"][spec.obs_key] = image

        state_parts = []
        for component in self.state_order:
            state = self._component_state(component)
            if state is None:
                return None
            state_parts.append(state)
        obs["arm_gripper_joints"] = (
            np.concatenate(state_parts) if state_parts else np.array([], dtype=np.float64)
        )
        return obs

    def publish_action(self, action: np.ndarray):
        if self._inference_start_time is not None:
            if time.time() - self._inference_start_time < self.warming_up_time:
                return
        targets = self._slice_action(action)
        self._publish_targets(targets)

    def _slice_action(self, action: np.ndarray) -> Dict[RobotComponent, np.ndarray]:
        targets: Dict[RobotComponent, np.ndarray] = {}
        idx = 0
        for component in self.action_order:
            size = _joint_count(self.robot_config, component)
            if idx + size > len(action):
                self.get_logger().warning(
                    f"Action vector too short for {component.name}: "
                    f"need {idx + size}, got {len(action)}"
                )
                break
            targets[component] = np.array(action[idx : idx + size], dtype=np.float64)
            idx += size
        return targets

    def _publish_targets(self, targets: Dict[RobotComponent, np.ndarray]):
        hand_targets: Dict[RobotComponent, np.ndarray] = {}
        direct_targets: Dict[RobotComponent, np.ndarray] = {}

        for component, values in targets.items():
            if component in (RobotComponent.LEFT_GRIPPER, RobotComponent.RIGHT_GRIPPER):
                hand_component, preset = self._resolve_gripper_action(component, float(values[0]))
                if hand_component is None:
                    continue
                hand_targets[hand_component] = preset
            else:
                direct_targets[component] = values

        for hand_component, preset in hand_targets.items():
            direct_targets[hand_component] = preset

        self.robot_io.publish_targets(direct_targets)

    def _resolve_gripper_action(
        self, component: RobotComponent, scalar: float
    ) -> Tuple[Optional[RobotComponent], np.ndarray]:
        cfg = self.gripper_config(component)
        threshold = float(cfg.get("threshold", self.policy_config.get("action", {}).get("gripper_threshold", 0.6)))
        closed = scalar >= threshold
        self.gripper_state[component] = 1.0 if closed else 0.0

        mode = cfg.get("command_mode", "hand_preset")
        if mode == "scalar_topic":
            topic_key = cfg.get("command_topic_key")
            if not topic_key:
                raise ValueError(f"{component.name} scalar_topic command missing command_topic_key")
            msg = self.Int32Msg()
            msg.data = int(cfg.get("close_value", 1) if closed else cfg.get("open_value", 0))
            pub = self.publishers.get(topic_key)
            if pub is None:
                raise ValueError(f"No publisher for gripper topic key: {topic_key}")
            pub.publish(msg)
            return None, np.array([], dtype=np.float64)

        hand_component = GRIPPER_TO_HAND[component]
        preset_key = "close_preset" if closed else "open_preset"
        preset = np.asarray(cfg[preset_key], dtype=np.float64)
        return hand_component, preset

    # ------------------------------------------------------------------
    # Reset and conditions
    # ------------------------------------------------------------------

    def warm_up(self):
        self.get_logger().info("Warming up, waiting for sensor data...")
        time.sleep(float(self.policy_config.get("runtime", {}).get("warmup_sleep", 1.0)))
        self.get_logger().info("Warm-up completed")

    def _initial_pose_components(self) -> Dict[RobotComponent, np.ndarray]:
        pose = self.policy_config.get("initial_pose") or {}
        result = {}
        for key, value in pose.items():
            component = _component_from_name(key)
            result[component] = np.asarray(value, dtype=np.float64)
        return result

    def _has_initial_pose(self) -> bool:
        return bool(self.policy_config.get("initial_pose"))

    def reset_home(self):
        targets = self._initial_pose_components()
        if not targets:
            self.get_logger().info("No initial pose set")
            return

        joint_targets = {
            comp: arr
            for comp, arr in targets.items()
            if comp
            in (
                RobotComponent.HEAD,
                RobotComponent.LEFT_ARM,
                RobotComponent.RIGHT_ARM,
                RobotComponent.LEFT_HAND,
                RobotComponent.RIGHT_HAND,
            )
        }
        gripper_targets = {
            comp: arr
            for comp, arr in targets.items()
            if comp in (RobotComponent.LEFT_GRIPPER, RobotComponent.RIGHT_GRIPPER)
        }

        ready_targets = {}
        deadline = time.time() + self._state_wait_timeout
        while time.time() < deadline:
            missing = [
                comp.name
                for comp in joint_targets
                if self._component_state(comp) is None
            ]
            if not missing:
                break
            time.sleep(0.05)
        for comp, target in joint_targets.items():
            current = self._component_state(comp)
            if current is None:
                self.get_logger().error(
                    f"reset_home skipped {comp.name}: state not received within "
                    f"{self._state_wait_timeout}s"
                )
                continue
            ready_targets[comp] = (current, target)

        if ready_targets:
            self.get_logger().info(
                f"Resetting initial pose ({self._reset_steps} steps, {self._reset_duration:.2f}s)"
            )
            dt = self._reset_duration / max(1, self._reset_steps)
            for step in range(self._reset_steps):
                alpha = float(step + 1) / float(self._reset_steps)
                step_targets = {
                    comp: current + alpha * (target - current)
                    for comp, (current, target) in ready_targets.items()
                }
                self._publish_targets(step_targets)
                time.sleep(dt)

        for comp, value in gripper_targets.items():
            self._publish_targets({comp: value.reshape(1)})
        self.get_logger().info("Initial pose reset completed")

    def _parse_complete_conditions_from_policy(self) -> List[Tuple[CompleteCondition, Any]]:
        raw = self.policy_config.get("complete_conditions", [])
        if not raw:
            duration = float(self.policy_config.get("duration", -1.0))
            return [(CompleteCondition.TIME_BASED, duration)]
        result = []
        for item in raw:
            condition_type = item.get("type", "TIME_BASED")
            condition = CompleteCondition[condition_type]
            if condition == CompleteCondition.TIME_BASED:
                result.append((condition, float(item.get("duration", -1.0))))
            elif condition == CompleteCondition.JOINT_REACHED:
                result.append(
                    (
                        condition,
                        JointReachedCondition(
                            target=item.get("target", {}),
                            threshold=float(item.get("threshold", 0.05)),
                            settle_time=float(item.get("settle_time", 0.3)),
                        ),
                    )
                )
            else:
                result.append((condition, item.get("param")))
        return result

    def setCompleteCondition(self, condition: CompleteCondition, param: Any = None):
        self.setCompleteConditions([(condition, param)])

    def setCompleteConditions(
        self,
        conditions: List[Union[Tuple[CompleteCondition, Any], CompleteCondition]],
    ):
        normalized = []
        for item in conditions:
            if isinstance(item, tuple):
                normalized.append(item)
            else:
                normalized.append((item, None))
        with self._state_lock:
            self._active_complete_conditions = normalized
            self._joint_reached_settle_starts = {}
        self.get_logger().info(
            f"Complete conditions: {[condition.name for condition, _ in normalized]}"
        )

    def _check_joint_reached(self, cond: JointReachedCondition) -> bool:
        parts = []
        for key, target in cond.target.items():
            if target is None:
                continue
            component = _component_from_name(key)
            current = self._component_state(component)
            if current is None:
                return False
            target_arr = np.asarray(target, dtype=object)
            if target_arr.ndim == 0:
                errors = np.array([abs(float(current[0]) - float(target_arr.item()))])
            else:
                if len(current) != len(target_arr):
                    return False
                errors = []
                for cur, tgt in zip(current, target_arr):
                    if tgt is None:
                        continue
                    errors.append(abs(float(cur) - float(tgt)))
                errors = np.asarray(errors, dtype=np.float64)
            if len(errors) and np.any(errors > cond.threshold):
                parts.append(False)
            else:
                parts.append(True)

        if not parts or not all(parts):
            self._joint_reached_settle_starts[id(cond)] = None
            with self._state_lock:
                self._condition_progress = 0.0
            return False

        now = time.time()
        if self._joint_reached_settle_starts.get(id(cond)) is None:
            self._joint_reached_settle_starts[id(cond)] = now
        elapsed = now - float(self._joint_reached_settle_starts[id(cond)])
        with self._state_lock:
            self._condition_progress = min(1.0, elapsed / cond.settle_time) if cond.settle_time > 0 else 1.0
        return elapsed >= cond.settle_time

    def _check_one_condition(self, condition: CompleteCondition, param: Any) -> bool:
        if condition == CompleteCondition.TIME_BASED:
            duration = float(param) if param is not None else 0.0
            if duration <= 0:
                return False
            elapsed = time.time() - self._start_time
            with self._state_lock:
                self._condition_progress = min(1.0, elapsed / duration)
            return elapsed >= duration
        if condition == CompleteCondition.JOINT_REACHED and isinstance(param, JointReachedCondition):
            return self._check_joint_reached(param)
        return False

    def _check_complete_condition(self) -> bool:
        eligible = self._complete_check_eligible_time
        if eligible is not None and time.time() < eligible:
            return False
        for condition, param in self._active_complete_conditions:
            if self._check_one_condition(condition, param):
                return True
        return False

    # ------------------------------------------------------------------
    # Public execution API
    # ------------------------------------------------------------------

    def start(
        self,
        timeout: float = -1.0,
        skip_warmup: bool = False,
        skip_reset: bool = False,
    ) -> bool:
        if not GeneralPolicyActionGenerator._global_execution_lock.acquire(blocking=False):
            self.get_logger().warning("Cannot start: another policy is running")
            return False
        try:
            GeneralPolicyActionGenerator._global_executing_instance = self.get_name()
            if not self._async_start_internal(skip_warmup=skip_warmup, skip_reset=skip_reset):
                return False
            if self._inference_thread:
                self._inference_thread.join(timeout=timeout if timeout > 0 else None)
                if timeout > 0 and self._inference_thread.is_alive():
                    self.get_logger().warning(f"Execution timeout ({timeout}s)")
                    self.stop()
                    return False
            return self.getProcessDone() and not self._stop_requested
        finally:
            GeneralPolicyActionGenerator._global_executing_instance = None
            if GeneralPolicyActionGenerator._global_execution_lock.locked():
                GeneralPolicyActionGenerator._global_execution_lock.release()

    def asyncStart(self, skip_warmup: bool = False, skip_reset: bool = False) -> bool:
        if not GeneralPolicyActionGenerator._global_execution_lock.acquire(blocking=False):
            self.get_logger().warning("Cannot start: another policy is running")
            return False
        with self._state_lock:
            if self._execution_running:
                GeneralPolicyActionGenerator._global_execution_lock.release()
                return False
        GeneralPolicyActionGenerator._global_executing_instance = self.get_name()
        return self._async_start_internal(skip_warmup=skip_warmup, skip_reset=skip_reset)

    def _async_start_internal(self, skip_warmup: bool = False, skip_reset: bool = False) -> bool:
        with self._state_lock:
            self._stop_requested = False
            self._process_done = False
            self._start_time = time.time()
            self._current_task_status = "Initializing"
            self._skip_warmup = skip_warmup
            self._skip_reset = skip_reset
            self._joint_reached_settle_starts = {}
            self._complete_check_eligible_time = None
            self._inference_count = 0
            self._condition_progress = 0.0
        with self.action_queue_lock:
            self.action_queue.clear()
            self.last_action = None
            self.actions_consumed_since_last_inference = 0
            self.last_n_consumed = 0
        if hasattr(self.action_policy, "reset") and callable(self.action_policy.reset):
            self.action_policy.reset()
        self._inference_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self._inference_thread.start()
        return True

    def stop(self):
        self.get_logger().info("Stop requested")
        with self._state_lock:
            self._stop_requested = True
            self._current_task_status = "Stopping"
        with self.action_queue_lock:
            self.action_queue.clear()
            self.last_action = None
        self.inference_running = False
        self.publish_running = False
        for thread in (self._inference_thread, self._publish_thread):
            if thread and thread.is_alive():
                thread.join(timeout=0.5)

    def getTaskStatus(self) -> str:
        with self._state_lock:
            return self._current_task_status

    def getProcessDone(self) -> bool:
        with self._state_lock:
            return self._process_done

    def getInferenceInfo(self) -> dict:
        with self._state_lock:
            return {
                "inference_count": self._inference_count,
                "action_queue_remaining": len(self.action_queue),
                "condition_progress": round(self._condition_progress, 3),
            }

    def _signal_handler(self, sig, frame):
        self.get_logger().info(f"Signal {sig} received, stopping execution")
        self.stop()
        sys.exit(0)

    # ------------------------------------------------------------------
    # Execution loop
    # ------------------------------------------------------------------

    def _execution_loop(self):
        try:
            with self._state_lock:
                self._execution_running = True
                self._process_done = False
                skip_warmup = getattr(self, "_skip_warmup", False)
                skip_reset = getattr(self, "_skip_reset", False)

            if not skip_warmup:
                with self._state_lock:
                    self._current_task_status = "Warming up"
                self.warm_up()
            if not skip_reset:
                with self._state_lock:
                    self._current_task_status = "Resetting home"
                self.reset_home()

            with self._state_lock:
                self._current_task_status = "Running"
                self._complete_check_eligible_time = time.time() + self._min_run_time_before_complete

            self._inference_start_time = time.time()
            self.inference_running = True
            self.publish_running = True
            inference_thread = threading.Thread(target=self._inference_thread_impl, daemon=True)
            publish_thread = threading.Thread(target=self._publish_thread_impl, daemon=True)
            self._publish_thread = publish_thread
            inference_thread.start()
            publish_thread.start()

            while not self._stop_requested:
                if self._check_complete_condition():
                    self.get_logger().info("Complete condition met")
                    break
                time.sleep(0.1)

            self.inference_running = False
            self.publish_running = False
            inference_thread.join(timeout=1.0)
            publish_thread.join(timeout=1.0)
        except Exception as exc:
            self.get_logger().error(f"Error in execution loop: {exc}")
            import traceback

            self.get_logger().error(traceback.format_exc())
        finally:
            with self._state_lock:
                self._execution_running = False
                self._process_done = True
                self._current_task_status = "Completed"
            self.inference_running = False
            self.publish_running = False
            if GeneralPolicyActionGenerator._global_execution_lock.locked():
                GeneralPolicyActionGenerator._global_executing_instance = None
                GeneralPolicyActionGenerator._global_execution_lock.release()

    def _inference_thread_impl(self):
        self.get_logger().info("Inference thread started")
        while self.inference_running:
            obs_start = time.time()
            obs = self.get_obs()
            obs_duration = time.time() - obs_start
            if obs is None:
                self.get_logger().warning("Inference: observations not ready")
                time.sleep(0.1)
                continue

            inference_start = time.time()
            actions_list = self.action_policy.inference_batch(obs, n_consumed=self.last_n_consumed)
            inference_duration = time.time() - inference_start
            with self.action_queue_lock:
                n_consumed_actual = self.actions_consumed_since_last_inference
                self.action_queue.clear()
                self.action_queue.extend(actions_list)
                self.last_action = actions_list[0].copy() if actions_list else None
                self.actions_consumed_since_last_inference = 0
                self.last_n_consumed = n_consumed_actual
                self.queue_updated.set()
            with self._state_lock:
                self._inference_count += 1
            self.get_logger().info(
                f"Inference | obs={obs_duration*1000:.1f}ms "
                f"inference={inference_duration*1000:.1f}ms "
                f"actions={len(actions_list)} consumed={n_consumed_actual}"
            )
            time.sleep(0.01)
        self.get_logger().info("Inference thread stopped")

    def _publish_thread_impl(self):
        self.get_logger().info("Publish thread started")
        current_index = 0
        while self.publish_running:
            publish_start = time.time()
            action = None
            with self.action_queue_lock:
                if self.queue_updated.is_set():
                    current_index = 0
                    self.queue_updated.clear()
                if len(self.action_queue) > 0:
                    if current_index < len(self.action_queue):
                        action = self.action_queue[current_index]
                        self.last_action = action.copy() if hasattr(action, "copy") else action
                        current_index += 1
                        self.actions_consumed_since_last_inference += 1
                    else:
                        action = self.last_action if self.last_action is not None else self.action_queue[-1]
                else:
                    action = self.last_action

            if action is not None and not self._stop_requested:
                try:
                    if not isinstance(action, np.ndarray):
                        action = np.asarray(action, dtype=np.float64)
                    self.publish_action(action)
                except Exception as exc:
                    self.get_logger().error(f"Error publishing action: {exc}")

            sleep_time = max(0.0, self.publish_period - (time.time() - publish_start))
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.get_logger().info("Publish thread stopped")


def _main():
    parser = argparse.ArgumentParser(description="General upper-body policy action generator")
    parser.add_argument("--robot", required=True, help="Robot I/O JSON config")
    parser.add_argument("--policy", required=True, help="Policy/model JSON config")
    parser.add_argument("--timeout", type=float, default=-1.0, help="Synchronous execution timeout")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--skip-reset", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate JSON and print vector mapping")
    args = parser.parse_args()

    configs = load_and_validate_configs(args.robot, args.policy)
    if args.dry_run:
        print(describe_mapping(configs))
        return 0

    if rclpy is None:
        raise RuntimeError("rclpy is required unless --dry-run is used") from _ROS_IMPORT_ERROR

    rclpy.init()
    generator = None
    executor = None
    try:
        generator = GeneralPolicyActionGenerator(args.robot, args.policy)
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(generator)
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        success = generator.start(
            timeout=args.timeout,
            skip_warmup=args.skip_warmup,
            skip_reset=args.skip_reset,
        )
        return 0 if success else 1
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=1.0)
        if generator is not None:
            generator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(_main())
