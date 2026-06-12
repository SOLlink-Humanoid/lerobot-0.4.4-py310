#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Policy inference adapter for LeRobot ACT checkpoint deployment.

The deployment code keeps the old PolicyAgent public API, while this module
loads policy weights and processor pipelines using the LeRobot processor layout:

    config.json
    model.safetensors
    policy_preprocessor.json
    policy_postprocessor.json

The imports are intentionally lazy/fallback-based so the same deployment file
can run on Python 3.10 LeRobot 0.4.x and Python 3.12 LeRobot 0.5.x layouts.
"""

from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

try:
    from lerobot.utils.constants import (
        OBS_STATE,
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
except ImportError:
    OBS_STATE = "observation.state"
    POLICY_PREPROCESSOR_DEFAULT_NAME = "policy_preprocessor"
    POLICY_POSTPROCESSOR_DEFAULT_NAME = "policy_postprocessor"


def _get_pretrained_config_class():
    try:
        from lerobot.configs import PreTrainedConfig
    except ImportError:
        from lerobot.configs.policies import PreTrainedConfig
    return PreTrainedConfig


def _install_act_only_policies_namespace() -> None:
    """Avoid importing optional policy packages when only ACT is needed.

    LeRobot 0.4.x executes ``lerobot.policies.__init__`` before importing
    submodules. That ``__init__`` imports optional policies such as GROOT, which
    can require heavy dependencies unrelated to ACT deployment. Installing a
    lightweight package module with the same search path lets Python import
    ``lerobot.policies.act.*`` and ``lerobot.policies.utils`` directly.
    """
    if "lerobot.policies" in sys.modules:
        return

    import lerobot

    lerobot_file = getattr(lerobot, "__file__", None)
    if lerobot_file is None:
        return

    policies_dir = Path(lerobot_file).resolve().parent / "policies"
    if not policies_dir.exists():
        return

    policies_pkg = types.ModuleType("lerobot.policies")
    policies_pkg.__file__ = str(policies_dir / "__init__.py")
    policies_pkg.__path__ = [str(policies_dir)]
    policies_pkg.__package__ = "lerobot.policies"
    sys.modules["lerobot.policies"] = policies_pkg
    setattr(lerobot, "policies", policies_pkg)


def _install_lightweight_processor_namespace() -> None:
    """Avoid importing tokenizer/robot processors when loading ACT processors."""
    if "lerobot.processor" in sys.modules:
        return

    import lerobot

    lerobot_file = getattr(lerobot, "__file__", None)
    if lerobot_file is None:
        return

    processor_dir = Path(lerobot_file).resolve().parent / "processor"
    if not processor_dir.exists():
        return

    processor_pkg = types.ModuleType("lerobot.processor")
    processor_pkg.__file__ = str(processor_dir / "__init__.py")
    processor_pkg.__path__ = [str(processor_dir)]
    processor_pkg.__package__ = "lerobot.processor"
    sys.modules["lerobot.processor"] = processor_pkg
    setattr(lerobot, "processor", processor_pkg)

    # Re-export the lightweight type aliases expected by lerobot.policies.utils
    # without executing lerobot.processor.__init__, which imports optional
    # tokenizer processors.
    from lerobot.processor.core import (  # type: ignore[import-not-found]
        EnvAction,
        EnvTransition,
        PolicyAction,
        RobotAction,
        RobotObservation,
        TransitionKey,
    )

    processor_pkg.EnvAction = EnvAction
    processor_pkg.EnvTransition = EnvTransition
    processor_pkg.PolicyAction = PolicyAction
    processor_pkg.RobotAction = RobotAction
    processor_pkg.RobotObservation = RobotObservation
    processor_pkg.TransitionKey = TransitionKey


def _install_training_config_stub() -> None:
    """PreTrainedPolicy imports TrainPipelineConfig, but deployment never uses it."""
    if "lerobot.configs.train" in sys.modules:
        return

    train_mod = types.ModuleType("lerobot.configs.train")

    class TrainPipelineConfig:
        def save_pretrained(self, *args, **kwargs):
            raise RuntimeError("TrainPipelineConfig is unavailable in ACT deployment mode.")

    train_mod.TrainPipelineConfig = TrainPipelineConfig
    sys.modules["lerobot.configs.train"] = train_mod


def _install_lerobot_act_deploy_shims() -> None:
    _install_act_only_policies_namespace()
    _install_lightweight_processor_namespace()
    _install_training_config_stub()


def _get_act_policy_class():
    _install_lerobot_act_deploy_shims()
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import LeRobot ACTPolicy. This deployment adapter only supports ACT, "
            "so optional policies should not be required. Check the missing ACT dependency in "
            "the original traceback."
        ) from exc
    return ACTPolicy


def _load_pretrained_processors(config, model_path: Path, device: torch.device):
    _install_lerobot_act_deploy_shims()
    try:
        # Import only processor pieces needed by default ACT checkpoints. Importing
        # lerobot.processor.__init__ pulls tokenizer processors and optional
        # transformers symbols that are not needed for ACT deployment.
        import lerobot.processor.batch_processor  # noqa: F401
        import lerobot.processor.device_processor  # noqa: F401
        import lerobot.processor.normalize_processor  # noqa: F401
        import lerobot.processor.rename_processor  # noqa: F401
        from lerobot.processor.pipeline import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import LeRobot processor pipeline. Check that minimal processor "
            "dependencies are installed for the selected LeRobot version."
        ) from exc

    device_override = {"device": str(device)}
    return (
        PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            overrides={"device_processor": device_override},
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        ),
        PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            overrides={"device_processor": {"device": "cpu"}},
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def _prepare_observation_for_inference(raw_obs, device):
    _install_lerobot_act_deploy_shims()
    try:
        from lerobot.policies.utils import prepare_observation_for_inference
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import LeRobot prepare_observation_for_inference. Check that the "
            "selected LeRobot version and its minimal inference dependencies are installed."
        ) from exc
    return prepare_observation_for_inference(
        raw_obs,
        device,
        task="",
        robot_type="humanoid_upper_body",
    )


class PolicyAgent:
    """Load a LeRobot ACT policy and run deployment inference."""

    CAMERA_CONFIGS: Tuple[Tuple[str, str, Tuple[int, int]], ...] = (
        ("camera_top", "observation.images.camera_top", (640, 360)),
        ("camera_head", "observation.images.camera_head", (640, 360)),
        ("camera_wrist", "observation.images.camera_wrist", (320, 240)),
    )

    def __init__(self, model_path: str):
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = self._get_device()
        self.config = self._load_config()
        self.policy = self._load_policy()
        self.preprocessor, self.postprocessor = self._load_processors()
        self.cnt = 0
        self._warned_temporal_ensemble_batch = False

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU is available. Device set to:", device)
        else:
            device = torch.device("cpu")
            print(f"GPU is not available. Device set to: {device}. Inference will be slower than on GPU.")
        return device

    def _require_latest_checkpoint_files(self) -> None:
        required_files = [
            "config.json",
            "model.safetensors",
            f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        ]
        missing = [name for name in required_files if not (self.model_path / name).exists()]
        if missing:
            raise FileNotFoundError(
                "Latest LeRobot policy checkpoint is incomplete. "
                f"Missing files in {self.model_path}: {missing}. "
                "Expected a latest-format checkpoint with config.json, model.safetensors, "
                "policy_preprocessor.json, and policy_postprocessor.json. "
                "For older LeRobot checkpoints, run the latest LeRobot migration script first."
            )

    def _load_config(self):
        self._require_latest_checkpoint_files()
        print(f"Loading policy config from: {self.model_path}")
        PreTrainedConfig = _get_pretrained_config_class()
        config = PreTrainedConfig.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        if config.type != "act":
            raise ValueError(
                f"This deployment adapter currently supports latest LeRobot ACT checkpoints only; "
                f"got policy type '{config.type}'."
            )
        config.device = str(self.device)
        return config

    def _load_policy(self):
        print(f"Loading policy weights from: {self.model_path}")
        policy_cls = _get_act_policy_class()
        policy = policy_cls.from_pretrained(
            self.model_path,
            config=self.config,
            local_files_only=True,
        )
        policy.eval()
        policy.to(self.device)
        return policy

    def _load_processors(self):
        return _load_pretrained_processors(self.config, self.model_path, self.device)

    def reset(self):
        """Reset policy and processor episode state."""
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        if hasattr(self.preprocessor, "reset"):
            self.preprocessor.reset()
        if hasattr(self.postprocessor, "reset"):
            self.postprocessor.reset()

    def generate_obs(self) -> Dict[str, Any]:
        """Generate simulated observation data for smoke tests."""
        return {
            "images": {
                "camera_head": (np.random.rand(360, 640, 3) * 255).astype(np.uint8),
                "camera_wrist": (np.random.rand(240, 320, 3) * 255).astype(np.uint8),
            },
            "qpos": np.random.randn(16),
            "arm_gripper_joints": np.random.randn(16),
        }

    def inference(self, obs: Optional[Dict[str, Any]]):
        """Return one postprocessed action tensor on CPU."""
        if obs is None:
            print("Using simulated observation data")
            obs = self.generate_obs()

        input_data = self.prepare_inference_obs(obs)
        with torch.inference_mode():
            processed_obs = self.preprocessor(input_data)
            action = self.policy.select_action(processed_obs)
            action = self.postprocessor(action)
        return action.detach().cpu()

    def inference_batch(self, obs: Optional[Dict[str, Any]], n_consumed=None) -> List[np.ndarray]:
        """Return a list of postprocessed action vectors.

        The old deployment loop expects a list of numpy action vectors. Latest
        LeRobot ACT exposes chunk inference through predict_action_chunk().
        """
        if obs is None:
            print("Using simulated observation data")
            obs = self.generate_obs()

        input_data = self.prepare_inference_obs(obs)
        with torch.inference_mode():
            processed_obs = self.preprocessor(input_data)
            if self.config.temporal_ensemble_coeff is not None:
                if not self._warned_temporal_ensemble_batch:
                    print(
                        "Warning: temporal_ensemble_coeff is enabled; returning a single "
                        "select_action() result instead of predict_action_chunk()."
                    )
                    self._warned_temporal_ensemble_batch = True
                action = self.policy.select_action(processed_obs)
                action = self.postprocessor(action).detach().cpu()
                return [self._to_numpy_action(action)]

            actions = self.policy.predict_action_chunk(processed_obs)
            actions = self.postprocessor(actions).detach().cpu()

        if actions.ndim == 3:
            actions = actions.squeeze(0)
        elif actions.ndim == 1:
            actions = actions.unsqueeze(0)
        return [self._to_numpy_action(action) for action in actions]

    def prepare_inference_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert deployment observations to latest LeRobot inference input."""
        raw_obs: Dict[str, Any] = {}
        images = obs.get("images", {})
        cameras_processed = 0

        for camera_key, obs_key, target_size in self.CAMERA_CONFIGS:
            if camera_key not in images:
                continue
            image = self._prepare_image(images[camera_key], target_size)
            raw_obs[obs_key] = image
            cameras_processed += 1

        if cameras_processed == 0:
            raise KeyError(f"No valid camera found in obs['images']. Available keys: {list(images.keys())}")

        qpos = np.asarray(obs["arm_gripper_joints"], dtype=np.float32)
        raw_obs[OBS_STATE] = qpos
        self.cnt += 1

        return _prepare_observation_for_inference(raw_obs, self.device)

    def _prepare_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(f"Camera image must be HWC, got shape {image.shape}")
        width, height = target_size
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, dsize=target_size)
        return image.astype(np.uint8, copy=False)

    def _to_numpy_action(self, action: torch.Tensor) -> np.ndarray:
        action = action.detach().cpu()
        if action.ndim > 1:
            action = action.squeeze(0)
        return action.numpy()
