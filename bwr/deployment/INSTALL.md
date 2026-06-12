# General Policy Deployment Install

이 폴더는 학습 코드를 빼고 policy 추론/ROS deploy에 필요한 파일만 모은 실행 묶음입니다.

## 최신 leRobot 설치가 필요한가?

필요합니다.

`bwr/deployment/action_policy.py`는 모델 로딩과 normalization을 직접 구현하지 않고, 최신 leRobot의 policy와 processor pipeline을 사용합니다.

```python
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
```

그래서 `action_policy.py` 파일만 복사해서는 충분하지 않습니다. 로봇 PC의 Python 환경에 최신 `lerobot` package가 설치되어 있거나, `PYTHONPATH`에 최신 leRobot 소스의 `src`가 잡혀 있어야 합니다.

이번 deployment adapter는 1차로 최신 leRobot ACT checkpoint를 대상으로 합니다. 모델 폴더에는 아래 파일이 있어야 합니다.

```text
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
```

## 1. Python 패키지 설치

deploy 폴더에서 실행합니다.

```bash
python3 -m pip install -r requirements.txt
```

CUDA GPU를 사용할 경우 `torch`, `torchvision`은 로봇의 CUDA 버전에 맞춰 PyTorch 공식 명령으로 먼저 설치하는 편이 안전합니다. 그 다음 `requirements.txt`를 설치하면 됩니다.

## 2. leRobot 설치

로봇 PC에 최신 leRobot 소스가 있다면:

```bash
cd /path/to/bwr-lerobot
python3 -m pip install -e .
```

전체 repo를 로봇에 계속 들고 다니기 싫다면, 개발 PC에서 wheel을 만든 뒤 로봇 PC에 wheel만 복사해서 설치할 수도 있습니다.

```bash
cd /path/to/bwr-lerobot
python3 -m pip install build
python3 -m build --wheel
```

그 다음 로봇 PC에서:

```bash
python3 -m pip install /path/to/lerobot-*.whl
```

## 3. ROS workspace source

ROS2와 로봇 message package는 pip로 설치하는 항목이 아닙니다. 실행 전에 로봇 ROS workspace를 source 해야 합니다.

agibotX2는 `aimdk_msgs`가 필요합니다.

```bash
source /path/to/agibot_ros2_ws/install/setup.bash
```

WalkerE는 `bodyctrl_msgs`가 필요합니다.

```bash
source /path/to/walker_ros2_ws/install/setup.bash
```

## 4. 설정 확인

```bash
python3 bwr/deployment/general_policy_action_generator.py \
  --robot bwr/deployment/configs/robot/agibotX2.json \
  --policy bwr/deployment/configs/policy/example_upper_body_policy.json \
  --dry-run
```

## 5. 실행

`bwr/deployment/configs/policy/example_upper_body_policy.json`의 `model_path`를 실제 `pretrained_model` 경로로 수정한 뒤 실행합니다.

```bash
python3 bwr/deployment/general_policy_action_generator.py \
  --robot bwr/deployment/configs/robot/agibotX2.json \
  --policy bwr/deployment/configs/policy/example_upper_body_policy.json
```
