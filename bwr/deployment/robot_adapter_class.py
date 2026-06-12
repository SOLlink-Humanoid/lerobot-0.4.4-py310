#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robot-specific ROS I/O adapters for general_policy_action_generator."""

from enum import IntFlag, auto
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


class RobotComponent(IntFlag):
    """Upper-body component flags shared by all robot adapters."""

    NONE = 0
    HEAD = auto()
    LEFT_ARM = auto()
    RIGHT_ARM = auto()
    LEFT_HAND = auto()
    RIGHT_HAND = auto()
    LEFT_GRIPPER = auto()
    RIGHT_GRIPPER = auto()
    CAMERA_HEAD = auto()
    CAMERA_HAND_LEFT = auto()
    CAMERA_HAND_RIGHT = auto()


CAMERA_COMPONENTS = (
    RobotComponent.CAMERA_HEAD
    | RobotComponent.CAMERA_HAND_LEFT
    | RobotComponent.CAMERA_HAND_RIGHT
)

GRIPPER_TO_HAND = {
    RobotComponent.LEFT_GRIPPER: RobotComponent.LEFT_HAND,
    RobotComponent.RIGHT_GRIPPER: RobotComponent.RIGHT_HAND,
}

COMPONENT_TO_KEY = {
    RobotComponent.HEAD: "head",
    RobotComponent.LEFT_ARM: "left_arm",
    RobotComponent.RIGHT_ARM: "right_arm",
    RobotComponent.LEFT_HAND: "left_hand",
    RobotComponent.RIGHT_HAND: "right_hand",
    RobotComponent.LEFT_GRIPPER: "left_gripper",
    RobotComponent.RIGHT_GRIPPER: "right_gripper",
    RobotComponent.CAMERA_HEAD: "camera_head",
    RobotComponent.CAMERA_HAND_LEFT: "camera_hand_left",
    RobotComponent.CAMERA_HAND_RIGHT: "camera_hand_right",
}

KEY_TO_COMPONENT = {
    "HEAD": RobotComponent.HEAD,
    "head": RobotComponent.HEAD,
    "LEFT_ARM": RobotComponent.LEFT_ARM,
    "left_arm": RobotComponent.LEFT_ARM,
    "RIGHT_ARM": RobotComponent.RIGHT_ARM,
    "right_arm": RobotComponent.RIGHT_ARM,
    "LEFT_HAND": RobotComponent.LEFT_HAND,
    "left_hand": RobotComponent.LEFT_HAND,
    "RIGHT_HAND": RobotComponent.RIGHT_HAND,
    "right_hand": RobotComponent.RIGHT_HAND,
    "LEFT_GRIPPER": RobotComponent.LEFT_GRIPPER,
    "left_gripper": RobotComponent.LEFT_GRIPPER,
    "RIGHT_GRIPPER": RobotComponent.RIGHT_GRIPPER,
    "right_gripper": RobotComponent.RIGHT_GRIPPER,
    "CAMERA_HEAD": RobotComponent.CAMERA_HEAD,
    "camera_head": RobotComponent.CAMERA_HEAD,
    "CAMERA_HAND_LEFT": RobotComponent.CAMERA_HAND_LEFT,
    "camera_hand_left": RobotComponent.CAMERA_HAND_LEFT,
    "CAMERA_HAND_RIGHT": RobotComponent.CAMERA_HAND_RIGHT,
    "camera_hand_right": RobotComponent.CAMERA_HAND_RIGHT,
}

CANONICAL_STATE_TOPIC_KEYS = [
    "head",
    "arm",
    "hand",
    "left_hand",
    "right_hand",
    "left_gripper",
    "right_gripper",
]

CANONICAL_COMMAND_TOPIC_KEYS = [
    "head",
    "arm",
    "hand",
    "left_hand",
    "right_hand",
    "left_gripper",
    "right_gripper",
]

CANONICAL_JOINT_KEYS = [
    "head",
    "arm",
    "left_arm",
    "right_arm",
    "hand",
    "left_hand",
    "right_hand",
]

CANONICAL_GRIPPER_KEYS = ["LEFT_GRIPPER", "RIGHT_GRIPPER"]


def has_component(flags: RobotComponent, component: RobotComponent) -> bool:
    return (flags & component) == component


def component_from_name(name: str) -> RobotComponent:
    if name in KEY_TO_COMPONENT:
        return KEY_TO_COMPONENT[name]
    upper = str(name).upper()
    if upper in KEY_TO_COMPONENT:
        return KEY_TO_COMPONENT[upper]
    raise ValueError(f"Unknown component: {name}")


def component_key(component: RobotComponent) -> str:
    return COMPONENT_TO_KEY[component]


def component_list(names: Iterable[str]) -> List[RobotComponent]:
    return [component_from_name(name) for name in names]


def component_names(components: Iterable[RobotComponent]) -> List[str]:
    return [component.name for component in components]


def joint_count(robot_config: Dict[str, Any], component: RobotComponent) -> int:
    if component in (RobotComponent.LEFT_GRIPPER, RobotComponent.RIGHT_GRIPPER):
        return 1
    if component in (
        RobotComponent.CAMERA_HEAD,
        RobotComponent.CAMERA_HAND_LEFT,
        RobotComponent.CAMERA_HAND_RIGHT,
    ):
        return 0
    key = component_key(component)
    joints = robot_config.get("joints", {})
    if key not in joints:
        raise ValueError(f"robot config missing joints.{key}")
    joint_cfg = joints[key]
    if "order" in joint_cfg:
        return len(joint_cfg["order"])
    if "ids" in joint_cfg:
        return len(joint_cfg["ids"])
    raise ValueError(f"robot config joints.{key} must define order or ids")


def topic_enabled(entry: Optional[Dict[str, Any]]) -> bool:
    return bool(entry and entry.get("enabled", True) and entry.get("topic"))


def validate_topic_entry(section: str, key: str, entry: Any) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"robot config topics.{section}.{key} must be an object")
    for required in ["enabled", "topic", "message_type"]:
        if required not in entry:
            raise ValueError(f"robot config topics.{section}.{key} missing key: {required}")
    if entry["enabled"] and not entry["topic"]:
        raise ValueError(f"robot config topics.{section}.{key}.topic is required when enabled=true")
    if entry["enabled"] and not entry["message_type"]:
        raise ValueError(
            f"robot config topics.{section}.{key}.message_type is required when enabled=true"
        )


def validate_robot_schema(robot: Dict[str, Any]) -> None:
    required_top_keys = [
        "robot_name",
        "adapter",
        "control",
        "qos",
        "pd_params",
        "topics",
        "joints",
        "grippers",
    ]
    for key in required_top_keys:
        if key not in robot:
            raise ValueError(f"robot config missing required key: {key}")

    topics = robot["topics"]
    for section in ["states", "commands", "cameras"]:
        if section not in topics:
            raise ValueError(f"robot config missing topics.{section}")

    for key in CANONICAL_STATE_TOPIC_KEYS:
        if key not in topics["states"]:
            raise ValueError(f"robot config missing topics.states.{key}")
        validate_topic_entry("states", key, topics["states"][key])
    for key in CANONICAL_COMMAND_TOPIC_KEYS:
        if key not in topics["commands"]:
            raise ValueError(f"robot config missing topics.commands.{key}")
        validate_topic_entry("commands", key, topics["commands"][key])
    for key in ["CAMERA_HEAD", "CAMERA_HAND_LEFT", "CAMERA_HAND_RIGHT"]:
        if key not in topics["cameras"]:
            raise ValueError(f"robot config missing topics.cameras.{key}")
        validate_topic_entry("cameras", key, topics["cameras"][key])

    for key in CANONICAL_JOINT_KEYS:
        if key not in robot["joints"]:
            raise ValueError(f"robot config missing joints.{key}")
    for key in CANONICAL_GRIPPER_KEYS:
        if key not in robot["grippers"]:
            raise ValueError(f"robot config missing grippers.{key}")


class RobotIOAdapter:
    """Base class for robot-specific ROS I/O."""

    adapter_name = ""

    def __init__(self, owner):
        self.owner = owner

    def load_message_types(self):
        raise NotImplementedError

    def setup_ros_interfaces(self):
        raise NotImplementedError

    def publish_targets(self, targets: Dict[RobotComponent, np.ndarray]):
        raise NotImplementedError


class WalkerEAdapter(RobotIOAdapter):
    """WalkerE ROS message and topic adapter."""

    adapter_name = "walkerE"

    def load_message_types(self):
        try:
            from bodyctrl_msgs.msg import CmdSetMotorPosition, MotorStatusMsg, SetMotorPosition
        except ImportError as exc:
            raise RuntimeError(
                "bodyctrl_msgs is required for the walkerE adapter. "
                "Source the WalkerE ROS2 workspace before running."
            ) from exc
        node = self.owner
        node.CmdSetMotorPosition = CmdSetMotorPosition
        node.MotorStatusMsg = MotorStatusMsg
        node.SetMotorPosition = SetMotorPosition

    def setup_ros_interfaces(self):
        node = self.owner
        topics = node.robot_config["topics"]
        states = topics["states"]
        commands = topics["commands"]

        arm_state_topic = states["arm"]["topic"]
        node.subscriptions.append(
            node.create_subscription(
                node.MotorStatusMsg,
                arm_state_topic,
                self._status_callback,
                10,
            )
        )
        node.get_logger().info(f"Subscribed walkerE upper body state: {arm_state_topic}")

        for gripper in (RobotComponent.LEFT_GRIPPER, RobotComponent.RIGHT_GRIPPER):
            cfg = node.gripper_config(gripper)
            if cfg.get("state_source") == "scalar_topic":
                state_key = cfg.get("state_topic_key")
                if state_key and topic_enabled(states.get(state_key)):
                    node.subscriptions.append(
                        node.create_subscription(
                            node.Int32Msg,
                            states[state_key]["topic"],
                            node.make_scalar_gripper_callback(gripper),
                            10,
                        )
                    )

        for component, key in (
            (RobotComponent.LEFT_HAND, "left_hand"),
            (RobotComponent.RIGHT_HAND, "right_hand"),
        ):
            if topic_enabled(states.get(key)):
                node.subscriptions.append(
                    node.create_subscription(
                        node.JointStateMsg,
                        states[key]["topic"],
                        node.make_hand_joint_state_callback(component),
                        10,
                    )
                )

        node.publishers["head"] = node.create_publisher(
            node.CmdSetMotorPosition, commands["head"]["topic"], 10
        )
        node.publishers["arm"] = node.create_publisher(
            node.CmdSetMotorPosition, commands["arm"]["topic"], 10
        )
        if topic_enabled(commands.get("left_hand")):
            node.publishers["left_hand"] = node.create_publisher(
                node.JointStateMsg, commands["left_hand"]["topic"], 10
            )
        if topic_enabled(commands.get("right_hand")):
            node.publishers["right_hand"] = node.create_publisher(
                node.JointStateMsg, commands["right_hand"]["topic"], 10
            )
        for key in ("left_gripper", "right_gripper"):
            if topic_enabled(commands.get(key)):
                node.publishers[key] = node.create_publisher(
                    node.Int32Msg, commands[key]["topic"], 10
                )

    def _status_callback(self, msg):
        node = self.owner
        pos_by_id = {}
        for status in getattr(msg, "status", []):
            pos_by_id[int(status.name)] = float(status.pos)

        for component in (RobotComponent.HEAD, RobotComponent.LEFT_ARM, RobotComponent.RIGHT_ARM):
            key = component_key(component)
            joint_cfg = node.robot_config["joints"].get(key)
            if not joint_cfg:
                continue
            ids = [int(v) for v in joint_cfg["ids"]]
            node.joint_state[component] = np.array(
                [pos_by_id.get(joint_id, 0.0) for joint_id in ids],
                dtype=np.float64,
            )

    def publish_targets(self, targets: Dict[RobotComponent, np.ndarray]):
        node = self.owner
        arm_values: List[Tuple[List[int], np.ndarray, Dict[str, Any]]] = []
        for component in (RobotComponent.LEFT_ARM, RobotComponent.RIGHT_ARM):
            if component in targets:
                key = component_key(component)
                joint_cfg = node.robot_config["joints"][key]
                arm_values.append(([int(v) for v in joint_cfg["ids"]], targets[component], joint_cfg))

        if arm_values:
            msg = node.CmdSetMotorPosition()
            msg.header = node.HeaderMsg()
            msg.header.stamp = node.get_clock().now().to_msg()
            for ids, values, joint_cfg in arm_values:
                self._append_motor_cmds(msg, ids, values, joint_cfg)
            node.publishers["arm"].publish(msg)

        if RobotComponent.HEAD in targets:
            joint_cfg = node.robot_config["joints"]["head"]
            msg = node.CmdSetMotorPosition()
            msg.header = node.HeaderMsg()
            msg.header.stamp = node.get_clock().now().to_msg()
            self._append_motor_cmds(
                msg, [int(v) for v in joint_cfg["ids"]], targets[RobotComponent.HEAD], joint_cfg
            )
            node.publishers["head"].publish(msg)

        for component, pub_key in (
            (RobotComponent.LEFT_HAND, "left_hand"),
            (RobotComponent.RIGHT_HAND, "right_hand"),
        ):
            if component in targets and pub_key in node.publishers:
                msg = node.JointStateMsg()
                msg.header.stamp = node.get_clock().now().to_msg()
                msg.name = list(node.robot_config["joints"][component_key(component)]["order"])
                msg.position = [float(v) for v in targets[component]]
                node.publishers[pub_key].publish(msg)

    def _append_motor_cmds(
        self,
        msg,
        ids: List[int],
        values: np.ndarray,
        joint_cfg: Dict[str, Any],
    ):
        node = self.owner
        speed = float(joint_cfg.get("speed", 1.0))
        current = float(joint_cfg.get("current", 5.0))
        for motor_id, value in zip(ids, values):
            cmd = node.SetMotorPosition()
            cmd.name = int(motor_id)
            cmd.pos = float(value)
            cmd.spd = speed
            cmd.cur = current
            msg.cmds.append(cmd)


class AgibotX2Adapter(RobotIOAdapter):
    """Agibot X2 ROS message and topic adapter."""

    adapter_name = "agibotX2"

    def load_message_types(self):
        try:
            from aimdk_msgs.msg import (
                HandCommand,
                HandCommandArray,
                JointCommand,
                JointCommandArray,
                JointStateArray,
            )
        except ImportError as exc:
            raise RuntimeError(
                "aimdk_msgs is required for the agibotX2 adapter. "
                "Source the Agibot X2 ROS2 workspace before running."
            ) from exc
        node = self.owner
        node.HandCommand = HandCommand
        node.HandCommandArray = HandCommandArray
        node.JointCommand = JointCommand
        node.JointCommandArray = JointCommandArray
        node.JointStateArray = JointStateArray

    def setup_ros_interfaces(self):
        node = self.owner
        topics = node.robot_config["topics"]
        states = topics["states"]
        commands = topics["commands"]
        sub_qos = node.make_qos(states.get("qos", node.robot_config.get("qos", {}).get("subscriber")))
        pub_qos = node.make_qos(commands.get("qos", node.robot_config.get("qos", {}).get("publisher")))
        hand_sub_qos = node.make_qos(
            states.get("hand_qos", node.robot_config.get("qos", {}).get("hand_subscriber"))
        )

        if topic_enabled(states.get("arm")):
            node.subscriptions.append(
                node.create_subscription(
                    node.JointStateArray,
                    states["arm"]["topic"],
                    self._arm_state_callback,
                    sub_qos,
                )
            )
        if topic_enabled(states.get("head")):
            node.subscriptions.append(
                node.create_subscription(
                    node.JointStateArray,
                    states["head"]["topic"],
                    self._head_state_callback,
                    sub_qos,
                )
            )
        if topic_enabled(states.get("hand")):
            node.subscriptions.append(
                node.create_subscription(
                    node.JointStateArray,
                    states["hand"]["topic"],
                    self._hand_state_callback,
                    hand_sub_qos,
                )
            )

        node.publishers["arm"] = node.create_publisher(
            node.JointCommandArray, commands["arm"]["topic"], pub_qos
        )
        node.publishers["head"] = node.create_publisher(
            node.JointCommandArray, commands["head"]["topic"], pub_qos
        )
        node.publishers["hand"] = node.create_publisher(
            node.HandCommandArray, commands["hand"]["topic"], 10
        )

    def _arm_state_callback(self, msg):
        node = self.owner
        order = node.robot_config["joints"]["arm"]["full_order"]
        pos_by_name = {}
        for idx, joint_state in enumerate(getattr(msg, "joints", [])):
            if idx < len(order):
                pos_by_name[order[idx]] = float(joint_state.position)
        for component in (RobotComponent.LEFT_ARM, RobotComponent.RIGHT_ARM):
            key = component_key(component)
            names = node.robot_config["joints"][key]["order"]
            node.joint_state[component] = np.array(
                [pos_by_name.get(name, 0.0) for name in names],
                dtype=np.float64,
            )

    def _head_state_callback(self, msg):
        node = self.owner
        names = node.robot_config["joints"]["head"]["order"]
        values = []
        for idx, _name in enumerate(names):
            if idx < len(getattr(msg, "joints", [])):
                values.append(float(msg.joints[idx].position))
            else:
                values.append(0.0)
        node.joint_state[RobotComponent.HEAD] = np.array(values, dtype=np.float64)

    def _hand_state_callback(self, msg):
        node = self.owner
        hand_order = node.robot_config["joints"]["hand"]["order"]
        left_values = []
        right_values = []
        for idx, joint_state in enumerate(getattr(msg, "joints", [])):
            if idx < len(hand_order):
                left_values.append(float(joint_state.position))
            elif idx < len(hand_order) * 2:
                right_values.append(float(joint_state.position))
            else:
                break
        if left_values:
            node.joint_state[RobotComponent.LEFT_HAND] = np.array(left_values, dtype=np.float64)
        if right_values:
            node.joint_state[RobotComponent.RIGHT_HAND] = np.array(right_values, dtype=np.float64)

    def publish_targets(self, targets: Dict[RobotComponent, np.ndarray]):
        if RobotComponent.HEAD in targets:
            self._publish_head(targets[RobotComponent.HEAD])

        if RobotComponent.LEFT_ARM in targets or RobotComponent.RIGHT_ARM in targets:
            self._publish_arm(
                left=targets.get(RobotComponent.LEFT_ARM),
                right=targets.get(RobotComponent.RIGHT_ARM),
            )

        if RobotComponent.LEFT_HAND in targets or RobotComponent.RIGHT_HAND in targets:
            self._publish_hand(
                left=targets.get(RobotComponent.LEFT_HAND),
                right=targets.get(RobotComponent.RIGHT_HAND),
            )

    def _make_joint_command(self, name: str, position: float, joint_cfg: Dict[str, Any]):
        node = self.owner
        cmd = node.JointCommand()
        cmd.name = name
        cmd.position = float(position)
        cmd.velocity = float(joint_cfg.get("velocity", 0.0))
        cmd.effort = float(joint_cfg.get("effort", 0.0))
        cmd.stiffness = float(
            joint_cfg.get("stiffness", node.robot_config.get("pd_params", {}).get("arm", {}).get("kp", 20.0))
        )
        cmd.damping = float(
            joint_cfg.get("damping", node.robot_config.get("pd_params", {}).get("arm", {}).get("kd", 2.0))
        )
        return cmd

    def _publish_head(self, values: np.ndarray):
        node = self.owner
        joint_cfg = node.robot_config["joints"]["head"]
        msg = node.JointCommandArray()
        for name, value in zip(joint_cfg["order"], values):
            msg.joints.append(self._make_joint_command(name, float(value), joint_cfg))
        node.publishers["head"].publish(msg)

    def _publish_arm(self, left: Optional[np.ndarray], right: Optional[np.ndarray]):
        node = self.owner
        full_order = node.robot_config["joints"]["arm"]["full_order"]
        left_names = node.robot_config["joints"]["left_arm"]["order"]
        right_names = node.robot_config["joints"]["right_arm"]["order"]
        left_current = self._state_or_zeros(RobotComponent.LEFT_ARM)
        right_current = self._state_or_zeros(RobotComponent.RIGHT_ARM)
        left_values = left if left is not None else left_current
        right_values = right if right is not None else right_current
        left_by_name = dict(zip(left_names, left_values))
        right_by_name = dict(zip(right_names, right_values))

        joint_cfg = node.robot_config["joints"]["arm"]
        msg = node.JointCommandArray()
        for name in full_order:
            if name in left_by_name:
                position = left_by_name[name]
            elif name in right_by_name:
                position = right_by_name[name]
            else:
                position = 0.0
            msg.joints.append(self._make_joint_command(name, float(position), joint_cfg))
        node.publishers["arm"].publish(msg)

    def _state_or_zeros(self, component: RobotComponent) -> np.ndarray:
        node = self.owner
        state = node.joint_state.get(component)
        if state is not None:
            return np.array(state, dtype=np.float64)
        return np.zeros(joint_count(node.robot_config, component), dtype=np.float64)

    def _publish_hand(self, left: Optional[np.ndarray], right: Optional[np.ndarray]):
        node = self.owner
        order = node.robot_config["joints"]["hand"]["order"]
        left_values = left if left is not None else self._state_or_zeros(RobotComponent.LEFT_HAND)
        right_values = right if right is not None else self._state_or_zeros(RobotComponent.RIGHT_HAND)
        hand_cfg = node.robot_config["joints"]["hand"]

        msg = node.HandCommandArray()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = hand_cfg.get("frame_id", "hand_command")
        msg.left_hand_type.value = int(hand_cfg.get("left_hand_type", 1))
        msg.right_hand_type.value = int(hand_cfg.get("right_hand_type", 1))
        msg.left_hands = [
            self._make_hand_command(f"left_{name}", float(pos), hand_cfg)
            for name, pos in zip(order, left_values)
        ]
        msg.right_hands = [
            self._make_hand_command(f"right_{name}", float(pos), hand_cfg)
            for name, pos in zip(order, right_values)
        ]
        node.publishers["hand"].publish(msg)

    def _make_hand_command(self, name: str, position: float, hand_cfg: Dict[str, Any]):
        node = self.owner
        cmd = node.HandCommand()
        cmd.name = name
        cmd.position = float(position)
        cmd.velocity = float(hand_cfg.get("velocity", 0.1))
        cmd.acceleration = float(hand_cfg.get("acceleration", 0.0))
        cmd.deceleration = float(hand_cfg.get("deceleration", 0.0))
        cmd.effort = float(hand_cfg.get("effort", 0.0))
        return cmd


ADAPTER_REGISTRY = {
    adapter.adapter_name: adapter
    for adapter in (WalkerEAdapter, AgibotX2Adapter)
}

SUPPORTED_ADAPTERS = tuple(ADAPTER_REGISTRY.keys())


def create_robot_adapter(owner) -> RobotIOAdapter:
    adapter_name = owner.robot_config["adapter"]
    try:
        adapter_cls = ADAPTER_REGISTRY[adapter_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported robot adapter: {adapter_name}") from exc
    return adapter_cls(owner)
