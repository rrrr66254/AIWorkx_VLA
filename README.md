# AIWorkx VLA - Mobile Game RL Pipeline

Vision-Language-Action (VLA) based autonomous mobile game playing pipeline.
Uses the NitroGen model as a teacher to train a Double DQN agent to play Subway Surfers.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Requirements](#2-requirements)
3. [Initial Server Setup](#3-initial-server-setup)
4. [Android Emulator Setup](#4-android-emulator-setup)
5. [NitroGen Installation](#5-nitrogen-installation)
6. [Code Deployment](#6-code-deployment)
7. [Running the Pipeline](#7-running-the-pipeline)
8. [Local Monitoring](#8-local-monitoring-viewerpy)
9. [File Structure and Roles](#9-file-structure-and-roles)
10. [RL Training Principles](#10-rl-training-principles)
11. [Key Coordinates and Pixel Detection](#11-key-coordinates-and-pixel-detection)
12. [Common Issues](#12-common-issues)

---

## 1. System Architecture

```
[Local PC (Windows)]
  config.py           -- your personal server connection info (not committed to git)
  server_setup.py     -- one-time server setup (SSH)
  deploy.py           -- code upload (SFTP)
  viewer.py           -- real-time monitoring GUI
        |
        | SSH / SFTP (paramiko)
        |
[Linux Server (RTX 5090 / RTX 4090 or better recommended)]
  Xvfb :1             -- virtual display (headless, no monitor needed)
  Android AVD         -- emulator (1080x2280)
  ~/vla_pipeline/
    pipeline_ng_rl.py -- main RL pipeline
    rl_agent.py       -- Double DQN agent
    fast_capture.py   -- scrcpy high-speed capture
    adb_env.py        -- ADB touch injection
    nitrogen_client.py-- NitroGen inference client
    ...
  watchdog_ng_rl.sh   -- auto-restart script
```

---

## 2. Requirements

### Local PC

- Python 3.10 or higher
- `pip install paramiko opencv-python numpy`

### Server (Linux)

- GPU: RTX 4090 / RTX 5090 or better recommended (minimum 16 GB VRAM)
- Ubuntu 20.04 / 22.04
- CUDA 12.x
- Python 3.10 or higher (Miniconda recommended)
- Items automatically installed by `server_setup.py`:
  - xvfb, adb, openjdk-17, git
  - Android SDK cmdline-tools, API 34 emulator
  - Miniconda, pip: torch, opencv-python-headless, numpy, scrcpy-client, pyzmq

---

## 3. Initial Server Setup

### 3-1. Create config.py

Each team member creates their own `config.py` from the provided template.
This file is listed in `.gitignore` and is never committed.

```bash
cp config.example.py config.py
```

Then open `config.py` and fill in your own values:

```python
SERVER_HOST = "YOUR_SERVER_IP"    # e.g. "192.168.1.100"
SERVER_PORT = 22
SERVER_USER = "YOUR_USERNAME"     # your Linux account on the server
SERVER_PASS = "YOUR_PASSWORD"

# All remote paths below are auto-derived from SERVER_USER.
# No changes needed unless your home directory is non-standard.
```

All remote paths (`~/android-sdk`, `~/vla_pipeline`, `~/NitroGen`, etc.)
are automatically derived from `SERVER_USER`, so you only need to set your
credentials.

Alternatively, use environment variables instead of editing the file:

```bash
export VLA_HOST=YOUR_SERVER_IP
export VLA_USER=YOUR_USERNAME
export VLA_PASS=YOUR_PASSWORD
```

### 3-2. Automated Server Setup

```bash
python server_setup.py
```

This script automatically:
1. Installs required apt packages (Xvfb, ADB, OpenJDK, git, ...)
2. Installs Miniconda + pip packages (torch, opencv, scrcpy-client, pyzmq)
3. Downloads Android SDK cmdline-tools
4. Downloads API 34 x86_64 system image
5. Creates AVD (GameTest)
6. Clones and installs NitroGen
7. Deploys pipeline code to `~/vla_pipeline/`
8. Creates helper scripts (start_emulator.sh, stop_emulator.sh, watchdog_ng_rl.sh)

Estimated time: approximately 15-20 minutes (depending on internet speed)

### 3-3. Verify Setup

After setup completes, start the emulator and open the viewer:

```bash
# On the server (SSH)
bash ~/vla_pipeline/start_emulator.sh

# On local PC
python viewer.py
```

Open `http://localhost:8080` — if the emulator screen appears, setup is successful.

---

## 4. Android Emulator Setup

`server_setup.py` creates the AVD automatically, but if manual setup is needed:

### 4-1. Manual Emulator Start (from server SSH)

```bash
export DISPLAY=:1
export ANDROID_SDK_ROOT=~/android-sdk

~/android-sdk/emulator/emulator \
  -avd GameTest \
  -no-window \
  -gpu swiftshader_indirect \
  -no-boot-anim \
  -no-audio &

adb wait-for-device
adb devices
```

### 4-2. APK Download and Installation

The APK is not included in this repository. Download it from APKPure or APKMirror.

**Recommended version: Subway Surfers 3.6.x (API 23+, x86_64)**

Download link (APKMirror — verifies APK signatures against Google Play originals):
- https://www.apkmirror.com/?s=subway+surfers

Look for: `com.kiloo.subwaysurf_X.XX.X-XXXXX_minAPI23(x86_64).apk`

**Transfer the APK to the server:**

```bash
# From local PC — copy APK to server via SCP
scp subway-surfers.apk YOUR_USERNAME@YOUR_SERVER_IP:~/

# Or use the SFTP helper (add to deploy.py or run manually)
python -c "
import paramiko
from config import SERVER_HOST, SERVER_PORT, SERVER_USER, SERVER_PASS, REMOTE_HOME
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(SERVER_HOST, SERVER_PORT, SERVER_USER, SERVER_PASS)
c.open_sftp().put('subway-surfers.apk', f'{REMOTE_HOME}/subway-surfers.apk')
print('Upload complete')
"
```

**Install on the emulator (run on server):**

```bash
# Make sure emulator is running first
~/android-sdk/platform-tools/adb -s emulator-5554 install ~/subway-surfers.apk

# Verify installation
~/android-sdk/platform-tools/adb shell pm list packages | grep subway
# Expected: package:com.kiloo.subwaysurf
```

### 4-3. First Game Launch and Setup

Several popups appear on the first launch. Use `viewer.py` (`http://localhost:8080`) to watch and interact with the screen:

1. Language selection -> English
2. Ad consent popup -> Accept or close with X
3. Login popup -> close with X
4. Daily Login popup -> close with BACK key
5. Confirm "Tap to Play" on the main screen

Only manual the first time; the pipeline handles it automatically from then on.

### 4-4. Verify Emulator Resolution

```bash
adb -s emulator-5554 shell wm size
# Output: Physical size: 1080x2280
```

If the resolution differs, update the pixel coordinates in `pipeline_ng_rl.py`.

---

## 5. NitroGen Installation

NitroGen is a VLA model that watches the game screen and predicts gamepad signals.

### 5-1. Install on Server

```bash
cd ~
git clone https://github.com/MineDojo/NitroGen
cd NitroGen
pip install -e ".[serve]"
```

### 5-2. Model Weights

Place the NitroGen model file (`ng.pt`) at `~/NitroGen/ng.pt`.

### 5-3. Start the Server

```bash
cd ~/NitroGen
python scripts/serve.py ng.pt --port 5556
```

Default port is 5556. Run in background:

```bash
nohup python scripts/serve.py ng.pt --port 5556 > ~/vla_pipeline/nitrogen.log 2>&1 &
```

### 5-4. Dummy Mode (testing without NitroGen)

To test only the pipeline structure without a NitroGen server:

```bash
python pipeline_ng_rl.py --dummy
```

In dummy mode, random gamepad output replaces NitroGen.

---

## 6. Code Deployment

Apply local code changes to the server.

```bash
# Upload all .py files inside vla_pipeline/
python deploy.py
```

`deploy.py` reads `SERVER_HOST`, `SERVER_USER`, `SERVER_PASS`, and `REMOTE_PROJECT_DIR`
from your `config.py` and uploads via SFTP — no hardcoded paths.

---

## 7. Running the Pipeline

### 7-1. Main Pipeline (NitroGen-guided RL)

Run directly from server SSH:

```bash
cd ~/vla_pipeline
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 DISPLAY=:1 \
  python pipeline_ng_rl.py \
  --device emulator-5554 \
  --step-interval 0.1 \
  --nitrogen-port 5556 \
  --no-record
```

#### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `--device` | auto-detect | ADB device serial (emulator-5554) |
| `--step-interval` | 0.3 | Step interval (seconds). 0.1 recommended |
| `--nitrogen-port` | 5556 | NitroGen server port |
| `--no-record` | False | Disable frame saving (saves CPU) |
| `--no-scrcpy` | False | Disable scrcpy, use ADB screencap |
| `--dummy` | False | Test with random actions without NitroGen |

### 7-2. Watchdog (auto-restart)

Running watchdog on the server automatically restarts the pipeline if it crashes:

```bash
nohup bash ~/vla_pipeline/watchdog_ng_rl.sh &
```

watchdog_ng_rl.sh contents (generated by `server_setup.py`, uses `$HOME` automatically):

```bash
#!/bin/bash
PYTHON="$HOME/miniconda3/bin/python3"
PIPELINE="$HOME/vla_pipeline/pipeline_ng_rl.py"
LOG="$HOME/vla_pipeline/pipeline_ng_rl.log"
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

### 7-3. Remote Execution from Local (via viewer.py)

Click the "Start NitroGen" button in `viewer.py` to start the pipeline via SSH from local.
All paths used by viewer.py are read from your `config.py`.

### 7-4. Sample Console Output

```
step=  345 | RL:RIGHT | e=0.268 | loss=0.0353 | buf=  316 | NG:34%|RL:66% | cap=21fps | dies=1
```

| Field | Meaning |
|-------|---------|
| `step` | Total environment steps |
| `RL:RIGHT` | Action selected this step (RL or NG) |
| `e=0.268` | Current epsilon (exploration rate) |
| `loss` | DQN Huber loss |
| `buf` | Replay Buffer size |
| `NG:34%/RL:66%` | NitroGen action ratio / RL action ratio |
| `cap=21fps` | scrcpy frame capture speed |
| `dies` | Game over count |

### 7-5. Expected Training Progress

| Time | Epsilon | Status |
|------|---------|--------|
| Just started | 1.0 | NitroGen plays 100% (data collection) |
| ~30 minutes | 0.7 | NitroGen 70%, RL 30% |
| ~1 hour | 0.5 | NitroGen 50%, RL 50% |
| ~3-4 hours | 0.1 | RL plays 90%+ (expected to surpass NitroGen) |

Checkpoint: `~/vla_pipeline/rl_ng_checkpoint.pt` (auto-saved every 200 steps)

---

## 8. Local Monitoring (viewer.py)

A real-time monitoring GUI that runs on the local PC.

```bash
python viewer.py
```

Open `http://localhost:8080` in your browser.

Displays:
- Emulator screen (live, refreshed every 200 ms)
- Pipeline log (live)
- Start / Stop NitroGen pipeline button

Controls in the browser:
- Click = tap on emulator
- Drag = swipe
- Right-click = long press

All SSH connection details and remote paths are read from `config.py`.

---

## 9. File Structure and Roles

```
AIWorkx_VLA/
  config.py               your personal server connection info (excluded from git)
  config.example.py       template -- copy to config.py and fill in your values
  server_setup.py         one-time server environment setup (SSH)
  deploy.py               uploads vla_pipeline/ code to server (reads from config.py)
  viewer.py               local monitoring web GUI (http://localhost:8080)
  grab_screen.py          capture current emulator screen (debug)
  status_check.py         quick pipeline status check (debug)
  game_setup.py           relaunch game + enter PLAY (emergency recovery)

  [pipeline files executed on server -- deployed via deploy.py]
  pipeline_ng_rl.py       NitroGen-guided RL main pipeline
  rl_agent.py             Double DQN agent (23-dim state)
  fast_capture.py         scrcpy high-speed frame capture (~20fps)
  pipeline_cnn.py         CNN DQN pipeline (alternative approach)
  rl_agent_cnn.py         CNN-based DQN agent
  action_mapper.py        NitroGen gamepad -> ADB swipe conversion (local copy)

  vla_pipeline/           server deployment target directory copy
    adb_env.py            ADB screen capture + touch injection
    action_mapper.py      NitroGen gamepad -> ADB touch conversion
    nitrogen_client.py    NitroGen inference client
    telemetry.py          CPU/GPU temperature, battery collection
    visualizer.py         draw action overlay on frames
    recorder.py           session recording (JSONL + PNG)
    pipeline.py           basic pipeline (no RL)
    requirements.txt      server pip package list
```

### Key File Details

#### adb_env.py

Handles all communication with the emulator via ADB.

- `capture_screen()`: captures screen -> returns numpy BGR array
- `execute(action_dict)`: executes tap / swipe / long_press / noop
- `get_screen_size()`: returns screen resolution
- `_run()`: executes ADB command (returns empty string on timeout, no exception)

#### rl_agent.py

Double DQN agent.

- State: NitroGen j_left (2-dim) + buttons (21-dim) = 23-dim float32 vector
- Actions: NOOP / LEFT / RIGHT / UP / DOWN (5 total)
- Reward: survival +0.1/step, game over -1.0
- Network: Linear(23->128->128->5), Double DQN, Huber Loss
- Buffer: Replay Buffer 20,000 entries

#### fast_capture.py

Uses scrcpy to provide ~20fps frame capture, faster than ADB screencap (~1fps).

- Based on the scrcpy-client Python package
- Continuously receives frames in a background thread
- `get_frame()`: immediately returns the latest frame (non-blocking)
- If scrcpy fails, `pipeline_ng_rl.py` automatically falls back to ADB screencap

#### nitrogen_client.py

Provides two modes:

- `DummyNitrogenClient`: random gamepad output (testing without NitroGen)
- `NitrogenServerClient`: communicates with NitroGen server via ZMQ, 256x256 frame input -> gamepad dict output

#### pipeline_ng_rl.py

Main training loop. Core logic:

```python
# With epsilon probability use NitroGen for exploration, with (1-epsilon) use RL
if random.random() < agent.epsilon:
    # exploration: use NitroGen's gamepad output as action
    ng_adb = mapper.map(nitrogen_raw)
    action_dict = ng_adb.to_dict()
    action_idx  = action_dict_to_idx(action_dict)
else:
    # exploitation: select the best action the RL has learned
    action_idx  = agent.select_action(state)
    action_dict = agent.get_action_dict(action_idx)
```

Every 30 steps, automatically checks if the game is in the foreground and relaunches if not.
Checkpoint is saved to `~/vla_pipeline/rl_ng_checkpoint.pt` (resolved at runtime via `os.path.expanduser`).

---

## 10. RL Training Principles

### NitroGen-guided Exploration

Standard DQN selects random actions during exploration.
In this pipeline, NitroGen's judgment is used as the exploration policy instead of random actions.

```
epsilon = 1.0  -> early stage: NitroGen plays 100% (collecting meaningful data)
epsilon = 0.5  -> mid stage: NitroGen 50% + RL 50%
epsilon = 0.1  -> late stage: RL plays 90%+ (goal is to surpass NitroGen)
```

NitroGen is not trained. Only the DQN agent is updated.

### State Representation

NitroGen watches the game screen and outputs gamepad signals.
Those output values (j_left 2D + buttons 21D = 23D) themselves become the RL state vector.
In other words, NitroGen's "game understanding" is used as the state.

### Game Over Detection

Game state is detected by specific pixel colors:

- Game over: pixel at (800, 2104) is green (g>140, b<80, g-r>60)
- Continue dialog: pixel at (540, 860) is blue (b>200, r<50)

---

## 11. Key Coordinates and Pixel Detection

Screen resolution: based on 1080x2280

| Purpose | Coordinates | Description |
|---------|-------------|-------------|
| Game over detection | (800, 2104) | Check for green pixel |
| Continue dialog | (540, 860) | Check for blue pixel |
| PLAY button | (810, 2220) | Main menu green button |
| IAP popup X | (1040, 248) | Close Permanent score boost etc. |
| Result screen X | (820, 175) | Close game result popup |
| Daily Login X | (1060, 460) | Close daily reward popup |
| Center tap | (540, 1200) | Game start confirmation tap |

On devices with different resolutions, coordinates must be adjusted proportionally.

---

## 12. Common Issues

### scrcpy Connection Failure

Symptom: `[ScrcpyCapture] timeout` or `cap=0fps`

```bash
# Verify scrcpy-client installation
pip install scrcpy-client --no-deps

# Check adbutils version compatibility (2.x required)
pip show adbutils

# Check ADB connection
adb devices
```

If scrcpy fails, the pipeline automatically falls back to ADB screencap (~1fps).

### Game Not in Foreground

Symptom: `dies=0` stays constant, screen moves to Google or Chrome

```bash
# Run game_setup.py from local (connects via SSH)
python game_setup.py
```

Or the pipeline's `ensure_game_foreground()` function handles this automatically every 30 steps.

### PLAY Button Blocked by Daily Login Popup

Symptom: game does not start, only popup keeps appearing

Use `viewer.py` (`http://localhost:8080`) to right-click (long press) or use the BACK key via SSH:

```bash
adb -s emulator-5554 shell input keyevent KEYCODE_BACK
adb -s emulator-5554 shell input keyevent KEYCODE_BACK
adb -s emulator-5554 shell input tap 810 2220
```

### Multiple Pipeline Instances Running

Symptom: duplicate logs, ADB command conflicts

```bash
# Kill all on server
pkill -9 -f pipeline_ng_rl.py
pkill -f watchdog_ng_rl.sh
# Restart single instance
bash ~/vla_pipeline/watchdog_ng_rl.sh &
```

### ADB Swipe Timeout

Symptom: `subprocess.TimeoutExpired: Command timed out after 10 seconds`

`_run()` in `adb_env.py` catches and handles TimeoutExpired.
On timeout it returns an empty string and the pipeline continues running.

### NitroGen Server Connection Failure

Symptom: `Connection refused` or `retrying connection`

```bash
# Check NitroGen server status
ps aux | grep serve.py

# Restart
cd ~/NitroGen
nohup python scripts/serve.py ng.pt --port 5556 > ~/vla_pipeline/nitrogen.log 2>&1 &

# Check port
ss -tlnp | grep 5556
```

### CUDA Out of Memory

Symptom: `CUDA OOM` error

```bash
# Check GPU usage
nvidia-smi

# Kill other CUDA processes then restart
CUDA_VISIBLE_DEVICES=0 python pipeline_ng_rl.py ...
```

---

## Additional Pipeline: CNN DQN (pipeline_cnn.py)

An alternative that learns directly from game screen pixels without NitroGen.

- 4-frame stack -> CNN -> Double DQN
- No NitroGen dependency
- Slower training and lower initial performance, but purely screen-based learning

```bash
python pipeline_cnn.py --device emulator-5554 --step-interval 0.3
```

Checkpoint: `~/vla_pipeline/rl_cnn_checkpoint.pt`

---

## Experiment Reproduction Summary

Steps for a team member to follow from scratch:

```
1. cp config.example.py config.py
   -- fill in SERVER_HOST, SERVER_USER, SERVER_PASS

2. python server_setup.py
   -- installs Xvfb, Android SDK, AVD, Miniconda, pip packages,
      NitroGen, pipeline code, and helper scripts (~15-20 min)

3. Download Subway Surfers APK (see Section 4-2) and transfer to server:
   scp subway-surfers.apk YOUR_USERNAME@YOUR_SERVER_IP:~/

4. On the server (SSH in):
   bash ~/vla_pipeline/start_emulator.sh
   ~/android-sdk/platform-tools/adb install ~/subway-surfers.apk

5. Launch the game once manually via viewer.py to dismiss first-run popups:
   -- python viewer.py  ->  http://localhost:8080
   -- Click through: language selection, ad consent, login popup, daily login

6. Place NitroGen weights on the server:
   -- Copy ng.pt to ~/NitroGen/ng.pt

7. python viewer.py  ->  http://localhost:8080
   Click "Start NitroGen" to begin training.
   (or on server: nohup bash ~/vla_pipeline/watchdog_ng_rl.sh &)
```
