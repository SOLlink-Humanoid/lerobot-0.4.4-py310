# General Policy Action Generator Tutorial

`general_policy_action_generator.py`는 로봇 설정 파일 1개와 정책 설정 파일 1개를 받아 실행합니다.
로봇별 ROS message 처리 코드는 `bwr/deployment/robot_adapter_class.py`에 분리되어 있습니다.

## 참고 구조

```text
bwr/deployment/general_policy_action_generator.py      # 공통 정책 실행 코드
bwr/deployment/robot_adapter_class.py                  # 로봇별 ROS I/O adapter class
bwr/deployment/configs/robot/agibotX2.json             # agibotX2 topic/joint/gripper 설정
bwr/deployment/configs/robot/walkerE.json              # WalkerE topic/joint/gripper 설정
bwr/deployment/configs/policy/example_upper_body_policy.json
```

설치 방법은 `bwr/deployment/z.doc/general_policy_action_generator_install.md` 또는 deploy 패키지 루트의 `INSTALL.md`를 참고하세요.

실행 흐름은 아래처럼 보면 됩니다.

```text
CLI --robot/--policy
  -> robot config, policy config 로드
  -> robot config의 "adapter" 이름으로 adapter class 선택
  -> observation/action vector를 policy config 순서대로 구성
  -> robot adapter가 실제 ROS message publish/subscribe 처리
```

`general_policy_action_generator.py`는 로봇별 message 구조를 직접 알지 않도록 유지합니다. 새 로봇을 추가할 때는 robot config를 같은 schema에 맞춰 추가하고, 필요한 ROS message 변환만 `robot_adapter_class.py`에 adapter class로 추가합니다.

## 실행 환경

로봇 PC에는 아래 항목이 준비되어 있어야 합니다.

- ROS2 Python 패키지: `rclpy`, `cv_bridge`, `sensor_msgs`, `std_msgs`
- 로봇 message 패키지: agibotX2는 `aimdk_msgs`, WalkerE는 `bodyctrl_msgs`
- Python 패키지: `numpy`, `opencv-python`, `torch`
- 최신 `lerobot` Python package

`action_policy.py`가 최신 leRobot의 `PreTrainedConfig`, `get_policy_class`, `make_pre_post_processors`를 사용하므로, 실행 환경에는 최신 `lerobot`이 설치되어 있거나 `PYTHONPATH`에 최신 leRobot 소스의 `src`가 잡혀 있어야 합니다. 모델 폴더에는 `config.json`, `model.safetensors`, `policy_preprocessor.json`, `policy_postprocessor.json`가 있어야 합니다.

## 1. 설정 파일 선택

로봇별 ROS topic 설정:

```bash
bwr/deployment/configs/robot/agibotX2.json
bwr/deployment/configs/robot/walkerE.json
```

정책/모델/observation/action 설정:

```bash
bwr/deployment/configs/policy/example_upper_body_policy.json
```

agibotX2를 사용할 때는 먼저 `bwr/deployment/configs/robot/agibotX2.json`의 카메라 topic을 실제 로봇 topic으로 수정하세요. 파일 안에 `"todo": "수정해서 사용하세요"`가 표시되어 있습니다.

## 2. 정책 설정 수정

`bwr/deployment/configs/policy/example_upper_body_policy.json`에서 최소한 아래를 수정합니다.

```json
{
  "model_path": "/path/to/your/pretrained_model",
  "observation": {
    "state_order": ["LEFT_ARM", "RIGHT_ARM", "LEFT_GRIPPER", "RIGHT_GRIPPER"]
  },
  "action": {
    "order": ["LEFT_ARM", "RIGHT_ARM", "LEFT_GRIPPER", "RIGHT_GRIPPER"]
  }
}
```

`state_order`와 `action.order`의 순서가 모델이 학습한 observation/action vector 순서와 같아야 합니다.

## 3. 실행 전 매핑 확인

ROS 없이 설정과 vector 순서만 확인하려면:

```bash
python3 bwr/deployment/general_policy_action_generator.py \
  --robot bwr/deployment/configs/robot/agibotX2.json \
  --policy bwr/deployment/configs/policy/example_upper_body_policy.json \
  --dry-run
```

## 4. 실행

ROS workspace를 source 한 뒤 실행합니다.

```bash
python3 bwr/deployment/general_policy_action_generator.py \
  --robot bwr/deployment/configs/robot/agibotX2.json \
  --policy bwr/deployment/configs/policy/example_upper_body_policy.json
```

WalkerE로 바꾸려면 `--robot` 경로만 바꾸면 됩니다.

```bash
python3 bwr/deployment/general_policy_action_generator.py \
  --robot bwr/deployment/configs/robot/walkerE.json \
  --policy bwr/deployment/configs/policy/example_upper_body_policy.json
```

새 로봇을 추가할 때는 `bwr/deployment/configs/robot/`에 robot config를 추가하고, `bwr/deployment/robot_adapter_class.py`에 adapter class를 하나 추가한 뒤 `ADAPTER_REGISTRY`에 등록하면 됩니다.
