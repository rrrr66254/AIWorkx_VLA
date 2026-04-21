# AIWorkx VLA - Mobile Game RL Pipeline

Vision-Language-Action (VLA) 기반 모바일 게임 자율 플레이 파이프라인.
NitroGen 모델을 교사로 삼아 Double DQN 에이전트가 Subway Surfers를 학습합니다.

---

## 목차

1. [시스템 구성](#1-시스템-구성)
2. [환경 요구사항](#2-환경-요구사항)
3. [서버 최초 설정](#3-서버-최초-설정)
4. [Android 에뮬레이터 설정](#4-android-에뮬레이터-설정)
5. [NitroGen 설치](#5-nitrogen-설치)
6. [코드 배포](#6-코드-배포)
7. [파이프라인 실행](#7-파이프라인-실행)
8. [로컬 모니터링](#8-로컬-모니터링-viewerpy)
9. [파일 구조 및 역할](#9-파일-구조-및-역할)
10. [RL 학습 원리](#10-rl-학습-원리)
11. [주요 좌표 및 픽셀 감지](#11-주요-좌표-및-픽셀-감지)
12. [자주 발생하는 문제](#12-자주-발생하는-문제)

---

## 1. 시스템 구성

```
[로컬 PC (Windows)]
  config.py           -- 서버 접속 정보
  server_setup.py     -- 서버 1회 세팅 (SSH)
  deploy.py           -- 코드 업로드 (SFTP)
  viewer.py           -- 실시간 모니터링 GUI
        |
        | SSH / SFTP (paramiko)
        |
[Linux 서버 (RTX 5090 / RTX 4090 이상 권장)]
  Xvfb :1             -- 가상 디스플레이
  TigerVNC :1         -- VNC 서버 (로컬에서 화면 확인)
  Android AVD         -- 에뮬레이터 (1080x2280)
  /home/sltrain/vla_pipeline/
    pipeline_ng_rl.py -- 메인 RL 파이프라인
    rl_agent.py       -- Double DQN 에이전트
    fast_capture.py   -- scrcpy 고속 캡처
    adb_env.py        -- ADB 터치 주입
    nitrogen_client.py-- NitroGen 추론 클라이언트
    ...
  watchdog_ng_rl.sh   -- 자동 재시작 스크립트
```

---

## 2. 환경 요구사항

### 로컬 PC

- Python 3.10 이상
- pip install paramiko opencv-python numpy

### 서버 (Linux)

- GPU: RTX 4090 / RTX 5090 이상 권장 (최소 16GB VRAM)
- Ubuntu 20.04 / 22.04
- CUDA 12.x
- Python 3.10 이상 (Miniconda 권장)
- 서버에서 `server_setup.py`가 자동 설치하는 항목:
  - xvfb, tigervnc-standalone-server, adb
  - Android SDK cmdline-tools, API 34 에뮬레이터
  - pip: opencv-python-headless, numpy, torch, scrcpy-client

---

## 3. 서버 최초 설정

### 3-1. config.py 작성

프로젝트 루트에 `config.py`를 만듭니다. (git에 올리지 않음)

```python
SERVER_HOST = "163.152.x.x"       # 서버 IP
SERVER_PORT = 22
SERVER_USER = "sltrain"
SERVER_PASS = "your_password"

VNC_PORT     = 5901
VNC_PASSWORD = "vncpass"
VNC_GEOMETRY = "1920x1080"
VNC_DISPLAY  = ":1"

REMOTE_SDK_DIR     = "/home/sltrain/android-sdk"
REMOTE_PROJECT_DIR = "/home/sltrain/vla_pipeline"
AVD_NAME           = "subway_avd"
AVD_DEVICE         = "pixel_6"
ANDROID_API        = "34"
SYSTEM_IMAGE       = "system-images;android-34;google_apis;x86_64"
```

### 3-2. 서버 자동 세팅

```bash
python server_setup.py
```

이 스크립트가 자동으로:
1. 필수 apt 패키지 설치 (Xvfb, TigerVNC, ADB, OpenJDK)
2. Android SDK cmdline-tools 다운로드
3. API 34 x86_64 시스템 이미지 다운로드
4. AVD (subway_avd) 생성
5. VNC 비밀번호 설정
6. Xvfb + VNC + 에뮬레이터 자동 시작 스크립트 생성
7. pip 패키지 설치

소요 시간: 약 10-20분 (인터넷 속도에 따라 다름)

### 3-3. VNC로 서버 화면 확인

서버 세팅 완료 후 RealVNC Viewer 또는 TigerVNC Viewer로 접속합니다.

```
주소: 서버IP:5901
비밀번호: config.py의 VNC_PASSWORD
```

에뮬레이터 화면이 보이면 정상입니다.

---

## 4. Android 에뮬레이터 설정

`server_setup.py`가 AVD를 자동 생성하지만, 수동으로 설정해야 할 경우:

### 4-1. 에뮬레이터 수동 실행 (서버 SSH에서)

```bash
export DISPLAY=:1
export ANDROID_SDK_ROOT=/home/sltrain/android-sdk

/home/sltrain/android-sdk/emulator/emulator \
  -avd subway_avd \
  -no-window \
  -gpu swiftshader_indirect \
  -no-boot-anim \
  -no-audio &

adb wait-for-device
adb devices
```

### 4-2. APK 설치

Subway Surfers APK (버전 3.61.1 권장):

```bash
adb -s emulator-5554 install subway-surfers-3-61-1.xapk
# 또는
adb -s emulator-5554 install com.kiloo.subwaysurf_3.62.0-90863_minAPI23.apk
```

### 4-3. 게임 최초 실행 및 설정

게임을 처음 실행하면 여러 팝업이 나타납니다. VNC로 보면서 수동으로 진행:

1. 언어 선택 → English
2. 광고 동의 팝업 → Accept 또는 X로 닫기
3. 로그인 팝업 → X로 닫기
4. Daily Login 팝업 → BACK 키로 닫기
5. 메인 화면에서 "Tap to Play" 확인

최초 1회만 수동 진행 후, 이후부터는 파이프라인이 자동 처리합니다.

### 4-4. 에뮬레이터 해상도 확인

```bash
adb -s emulator-5554 shell wm size
# 출력: Physical size: 1080x2280
```

해상도가 다르면 `pipeline_ng_rl.py`의 픽셀 좌표를 수정해야 합니다.

---

## 5. NitroGen 설치

NitroGen은 게임 화면을 보고 gamepad 신호를 예측하는 VLA 모델입니다.

### 5-1. 서버에서 설치

```bash
cd /home/sltrain
git clone https://github.com/MineDojo/NitroGen
cd NitroGen
pip install -e ".[serve]"
```

### 5-2. 모델 가중치

NitroGen 모델 파일(`ng.pt`)을 `/home/sltrain/NitroGen/` 에 위치시킵니다.

### 5-3. 서버 실행

```bash
cd /home/sltrain/NitroGen
python scripts/serve.py ng.pt --port 5556
```

기본 포트는 5556입니다. 백그라운드 실행:

```bash
nohup python scripts/serve.py ng.pt --port 5556 > nitrogen.log 2>&1 &
```

### 5-4. 더미 모드 (NitroGen 없이 테스트)

NitroGen 서버 없이 파이프라인 구조만 테스트할 때:

```bash
python pipeline_ng_rl.py --dummy
```

더미 모드에서는 랜덤 gamepad 출력이 NitroGen을 대신합니다.

---

## 6. 코드 배포

로컬에서 코드를 수정했을 때 서버에 반영합니다.

```bash
# vla_pipeline/ 내 모든 .py 파일 업로드
python deploy.py

# 개별 파일 업로드 (server 디렉토리에 직접)
python final_setup.py
```

---

## 7. 파이프라인 실행

### 7-1. 메인 파이프라인 (NitroGen-guided RL)

서버 SSH에서 직접 실행:

```bash
cd /home/sltrain/vla_pipeline
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 DISPLAY=:1 \
  python pipeline_ng_rl.py \
  --device emulator-5554 \
  --step-interval 0.1 \
  --nitrogen-port 5556 \
  --no-record
```

#### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--device` | 자동 감지 | ADB 기기 시리얼 (emulator-5554) |
| `--step-interval` | 0.3 | 스텝 간격 (초). 0.1 권장 |
| `--nitrogen-port` | 5556 | NitroGen 서버 포트 |
| `--no-record` | False | 프레임 저장 비활성화 (CPU 절약) |
| `--no-scrcpy` | False | scrcpy 비활성화, ADB screencap 사용 |
| `--dummy` | False | NitroGen 없이 랜덤 액션으로 테스트 |

### 7-2. Watchdog (자동 재시작)

서버에서 watchdog을 함께 실행하면 파이프라인이 죽어도 자동 재시작합니다:

```bash
# watchdog_ng_rl.sh를 백그라운드로 실행
nohup bash /home/sltrain/vla_pipeline/watchdog_ng_rl.sh &
```

watchdog_ng_rl.sh 내용:

```bash
#!/bin/bash
PYTHON="/home/sltrain/miniconda3/bin/python3"
PIPELINE="/home/sltrain/vla_pipeline/pipeline_ng_rl.py"
LOG="/home/sltrain/vla_pipeline/pipeline_ng_rl.log"
DEVICE="emulator-5554"

while true; do
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 DISPLAY=:1 \
        $PYTHON -u $PIPELINE \
        --device $DEVICE \
        --step-interval 0.1 \
        --nitrogen-port 5556 \
        --no-record >> $LOG 2>&1
    echo "[watchdog] restart $(date)" >> $LOG
    sleep 5
done
```

### 7-3. 로컬에서 원격 실행 (viewer.py 통해)

`viewer.py`의 "Start Pipeline" 버튼을 클릭하면 로컬에서 SSH로 파이프라인을 실행합니다.

### 7-4. 콘솔 출력 예시

```
step=  345 | RL:RIGHT | e=0.268 | loss=0.0353 | buf=  316 | NG:34%|RL:66% | cap=21fps | dies=1
```

| 항목 | 의미 |
|------|------|
| `step` | 총 환경 스텝 수 |
| `RL:RIGHT` | 이번 스텝에서 선택된 액션 (RL 또는 NG) |
| `e=0.268` | 현재 epsilon (탐험률) |
| `loss` | DQN Huber 손실 |
| `buf` | Replay Buffer 크기 |
| `NG:34%/RL:66%` | NitroGen 액션 비율 / RL 액션 비율 |
| `cap=21fps` | scrcpy 프레임 캡처 속도 |
| `dies` | 게임 오버 횟수 |

### 7-5. 학습 진행 예상

| 시간 | epsilon | 상태 |
|------|---------|------|
| 시작 직후 | 1.0 | NitroGen이 100% 플레이 (데이터 수집) |
| 약 30분 | 0.7 | NitroGen 70%, RL 30% |
| 약 1시간 | 0.5 | NitroGen 50%, RL 50% |
| 약 3-4시간 | 0.1 | RL이 90% 이상 플레이 (NitroGen 초과 기대) |

체크포인트: `/home/sltrain/vla_pipeline/rl_ng_checkpoint.pt` (200스텝마다 자동 저장)

---

## 8. 로컬 모니터링 (viewer.py)

로컬 PC에서 실행하는 실시간 모니터링 GUI입니다.

```bash
python viewer.py
```

표시 항목:
- 서버 SSH 연결 상태
- NitroGen 서버 상태
- 파이프라인 실행 상태 (epsilon, steps, dies, fps)
- 로그 실시간 스트리밍
- Start / Stop 버튼

VNC로 에뮬레이터 화면도 함께 보면 학습 현황을 시각적으로 확인할 수 있습니다.

---

## 9. 파일 구조 및 역할

```
AIWorkx_VLA/
  config.py               로컬 서버 접속 정보 (git 제외)
  server_setup.py         서버 1회 환경 세팅 (SSH)
  deploy.py               vla_pipeline/ 코드를 서버에 업로드
  viewer.py               로컬 모니터링 GUI (tkinter)
  grab_screen.py          현재 에뮬레이터 화면 캡처 (디버그용)
  status_check.py         파이프라인 상태 빠른 확인 (디버그용)
  game_setup.py           게임 재실행 + PLAY 진입 (긴급 복구용)

  [서버에서 실행되는 파이프라인 파일]
  pipeline_ng_rl.py       NitroGen-guided RL 메인 파이프라인
  rl_agent.py             Double DQN 에이전트 (23차원 상태)
  fast_capture.py         scrcpy 고속 프레임 캡처 (~20fps)
  pipeline_cnn.py         CNN DQN 파이프라인 (대안 방식)
  rl_agent_cnn.py         CNN 기반 DQN 에이전트
  action_mapper.py        NitroGen gamepad -> ADB swipe 변환 (로컬 사본)

  vla_pipeline/           서버 배포 대상 디렉토리 사본
    adb_env.py            ADB 화면 캡처 + 터치 주입
    action_mapper.py      NitroGen gamepad -> ADB 터치 변환
    nitrogen_client.py    NitroGen 추론 클라이언트
    telemetry.py          CPU/GPU 온도, 배터리 수집
    visualizer.py         프레임에 액션 오버레이 그리기
    recorder.py           세션 기록 (JSONL + PNG)
    pipeline.py           기본 파이프라인 (RL 없음)
    requirements.txt      서버 pip 패키지 목록
```

### 핵심 파일 상세 설명

#### adb_env.py

에뮬레이터와 ADB로 통신하는 모든 기능을 담당합니다.

- `capture_screen()`: 화면 캡처 -> numpy BGR 배열 반환
- `execute(action_dict)`: tap / swipe / long_press / noop 실행
- `get_screen_size()`: 화면 해상도 반환
- `_run()`: ADB 명령 실행 (타임아웃 시 예외 없이 빈 문자열 반환)

#### rl_agent.py

Double DQN 에이전트입니다.

- 상태(State): NitroGen의 j_left(2차원) + buttons(21차원) = 23차원 float32 벡터
- 행동(Action): NOOP / LEFT / RIGHT / UP / DOWN (5개)
- 보상(Reward): 생존 +0.1/스텝, 게임 오버 -1.0
- 네트워크: Linear(23->128->128->5), Double DQN, Huber Loss
- 버퍼: Replay Buffer 20,000개

#### fast_capture.py

scrcpy를 사용해 ADB screencap(~1fps)보다 빠른 ~20fps 프레임 캡처를 제공합니다.

- scrcpy-client Python 패키지 기반
- 백그라운드 스레드로 프레임을 지속적으로 수신
- `get_frame()`: 최신 프레임 즉시 반환 (논블로킹)
- scrcpy 실패 시 `pipeline_ng_rl.py`가 자동으로 ADB screencap으로 폴백

#### nitrogen_client.py

두 가지 모드를 제공합니다:

- `DummyNitrogenClient`: 랜덤 gamepad 출력 (NitroGen 없이 테스트)
- `NitrogenServerClient`: ZMQ로 NitroGen 서버와 통신, 256x256 프레임 입력 -> gamepad dict 출력

#### pipeline_ng_rl.py

메인 학습 루프입니다. 핵심 로직:

```python
# epsilon 확률로 NitroGen 탐험, (1-epsilon) 확률로 RL 활용
if random.random() < agent.epsilon:
    # 탐험: NitroGen의 gamepad 출력을 액션으로 사용
    ng_adb = mapper.map(nitrogen_raw)
    action_dict = ng_adb.to_dict()
    action_idx  = action_dict_to_idx(action_dict)
else:
    # 활용: RL이 학습한 최선의 액션 선택
    action_idx  = agent.select_action(state)
    action_dict = agent.get_action_dict(action_idx)
```

30스텝마다 게임이 포그라운드에 있는지 자동 확인하고, 아니면 재실행합니다.

---

## 10. RL 학습 원리

### NitroGen-guided Exploration

일반 DQN은 탐험(exploration) 시 랜덤 액션을 선택합니다.
이 파이프라인에서는 랜덤 대신 NitroGen의 판단을 탐험 정책으로 사용합니다.

```
epsilon = 1.0  -> 초반: NitroGen이 100% 플레이 (의미 있는 데이터 수집)
epsilon = 0.5  -> 중반: NitroGen 50% + RL 50%
epsilon = 0.1  -> 후반: RL이 90% 이상 (NitroGen을 초과하는 것이 목표)
```

NitroGen은 학습되지 않습니다. 오직 DQN 에이전트만 업데이트됩니다.

### 상태 표현

NitroGen은 게임 화면을 보고 gamepad 신호를 출력합니다.
그 출력값(j_left 2D + buttons 21D = 23D) 자체가 RL의 상태 벡터가 됩니다.
즉, NitroGen의 "게임 이해도"를 상태로 활용합니다.

### 게임 오버 감지

특정 픽셀 색상으로 게임 상태를 감지합니다:

- 게임 오버: (800, 2104) 픽셀이 초록색 (g>140, b<80, g-r>60)
- Continue 다이얼로그: (540, 860) 픽셀이 파란색 (b>200, r<50)

---

## 11. 주요 좌표 및 픽셀 감지

화면 해상도: 1080x2280 기준

| 용도 | 좌표 | 설명 |
|------|------|------|
| 게임 오버 감지 | (800, 2104) | 초록 픽셀 확인 |
| Continue 다이얼로그 | (540, 860) | 파란 픽셀 확인 |
| PLAY 버튼 | (810, 2220) | 메인 메뉴 초록 버튼 |
| IAP 팝업 X | (1040, 248) | Permanent score boost 등 닫기 |
| 결과화면 X | (820, 175) | 게임 결과 팝업 닫기 |
| Daily Login X | (1060, 460) | 일일 보상 팝업 닫기 |
| 중앙 탭 | (540, 1200) | 게임 시작 확인 탭 |

해상도가 다른 기기에서는 좌표를 비례적으로 조정해야 합니다.

---

## 12. 자주 발생하는 문제

### scrcpy 연결 실패

증상: `[ScrcpyCapture] 타임아웃` 또는 `cap=0fps`

```bash
# scrcpy-client 설치 확인
pip install scrcpy-client --no-deps

# adbutils 버전 호환 확인 (2.x 필요)
pip show adbutils

# ADB 연결 확인
adb devices
```

scrcpy 실패 시 파이프라인이 자동으로 ADB screencap(~1fps)으로 폴백합니다.

### 게임이 포그라운드에 없을 때

증상: `dies=0`이 계속 유지, 화면이 Google 또는 Chrome으로 이동

```bash
# 서버에서 game_setup.py 실행
python game_setup.py
```

또는 파이프라인의 `ensure_game_foreground()` 함수가 30스텝마다 자동 처리합니다.

### Daily Login 팝업으로 PLAY 버튼이 막힐 때

증상: 게임이 실행되지 않고 팝업만 계속 뜸

VNC로 서버 화면에 접속해 BACK 키를 2-3회 누릅니다:

```bash
adb -s emulator-5554 shell input keyevent KEYCODE_BACK
adb -s emulator-5554 shell input keyevent KEYCODE_BACK
adb -s emulator-5554 shell input tap 810 2220
```

### 파이프라인이 여러 개 실행 중

증상: 로그가 중복, ADB 명령 충돌

```bash
# 서버에서 모두 종료
pkill -9 -f pipeline_ng_rl.py
pkill -f watchdog_ng_rl.sh
# 단일 인스턴스 재시작
bash watchdog_ng_rl.sh &
```

### ADB 스와이프 타임아웃

증상: `subprocess.TimeoutExpired: Command timed out after 10 seconds`

`adb_env.py`의 `_run()`이 TimeoutExpired를 잡아 처리합니다.
수정된 버전에서는 타임아웃 시 빈 문자열을 반환하고 파이프라인이 계속 실행됩니다.

### NitroGen 서버 연결 실패

증상: `Connection refused` 또는 `연결 재시도`

```bash
# NitroGen 서버 상태 확인
ps aux | grep serve.py

# 재실행
cd /home/sltrain/NitroGen
nohup python scripts/serve.py ng.pt --port 5556 > nitrogen.log 2>&1 &

# 포트 확인
ss -tlnp | grep 5556
```

### CUDA out of memory

증상: `CUDA OOM` 오류

```bash
# GPU 사용량 확인
nvidia-smi

# 다른 CUDA 프로세스 종료 후 재시작
CUDA_VISIBLE_DEVICES=0 python pipeline_ng_rl.py ...
```

---

## 추가 파이프라인: CNN DQN (pipeline_cnn.py)

NitroGen 없이 게임 화면 픽셀을 직접 학습하는 대안입니다.

- 4-frame 스택 -> CNN -> Double DQN
- NitroGen 의존성 없음
- 학습 속도가 느리고 초기 성능이 낮지만, 순수 화면 기반 학습

```bash
python pipeline_cnn.py --device emulator-5554 --step-interval 0.3
```

체크포인트: `rl_cnn_checkpoint.pt`

---

## 실험 재현 절차 요약

팀원이 처음부터 따라하는 순서:

```
1. config.py 작성 (서버 IP/계정 정보 입력)
2. python server_setup.py        -- 서버 환경 자동 구성 (~10분)
3. VNC로 서버 접속 확인
4. 에뮬레이터에 APK 설치 후 게임 최초 실행 (수동)
5. NitroGen 서버 실행 (서버 SSH에서)
6. python deploy.py              -- 최신 코드 서버에 업로드
7. python viewer.py              -- 로컬 모니터링 GUI 실행
8. viewer.py에서 "Start Pipeline" 클릭
   또는 서버에서 직접: bash watchdog_ng_rl.sh &
9. VNC + viewer.py 로 학습 진행 모니터링
```
