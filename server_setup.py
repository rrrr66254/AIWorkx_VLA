"""
서버 1회 세팅 스크립트.
로컬 PC에서 실행 → paramiko로 서버에 접속 → 필요한 패키지/환경 자동 설치.
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import paramiko
from config import (
    SERVER_HOST, SERVER_PORT, SERVER_USER, SERVER_PASS,
    VNC_PASSWORD, VNC_GEOMETRY, VNC_DISPLAY,
    REMOTE_SDK_DIR, AVD_NAME, AVD_DEVICE, ANDROID_API, SYSTEM_IMAGE,
    REMOTE_PROJECT_DIR,
)

ANDROID_CMDTOOLS_URL = (
    "https://dl.google.com/android/repository/"
    "commandlinetools-linux-11076708_latest.zip"
)


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(SERVER_HOST, SERVER_PORT, SERVER_USER, SERVER_PASS, timeout=15)
    return c


def run(c, cmd, timeout=60, show=True):
    _, out, err = c.exec_command(cmd, timeout=timeout)
    stdout = out.read().decode("utf-8", errors="ignore").strip()
    stderr = err.read().decode("utf-8", errors="ignore").strip()
    if show and stdout:
        print(f"  → {stdout[:300]}")
    if stderr and "warning" not in stderr.lower():
        print(f"  [stderr] {stderr[:200]}")
    return stdout


def step(msg):
    print(f"\n{'─'*55}\n▶ {msg}\n{'─'*55}")


def setup_system(c):
    step("1. 시스템 패키지 설치")
    pkgs = (
        "xvfb tigervnc-standalone-server tigervnc-common "
        "adb python3-pip python3-venv "
        "openjdk-17-jdk-headless unzip wget curl libgl1"
    )
    run(c, f"sudo DEBIAN_FRONTEND=noninteractive apt-get install -y {pkgs}", timeout=180)
    print("  ✓ 패키지 설치 완료")


def setup_vnc(c):
    step("2. VNC 설정")
    run(c, "mkdir -p ~/.vnc")
    # 비밀번호 설정 (non-interactive)
    run(c, f"printf '{VNC_PASSWORD}\\n{VNC_PASSWORD}\\nn\\n' | vncpasswd", timeout=10)
    # xstartup 파일 생성
    xstartup = "#!/bin/sh\\nexec openbox-session &"
    run(c, f"echo '{xstartup}' > ~/.vnc/xstartup && chmod +x ~/.vnc/xstartup", timeout=5)
    # openbox 설치 (가벼운 WM)
    run(c, "sudo apt-get install -y openbox", timeout=60)
    print("  ✓ VNC 설정 완료")


def setup_android_sdk(c):
    step("3. Android SDK 설치")
    # 이미 설치됐으면 스킵
    if "cmdline-tools" in run(c, f"ls {REMOTE_SDK_DIR}/cmdline-tools 2>/dev/null || echo ''", show=False):
        print("  ✓ 이미 설치되어 있음, 스킵")
        return

    run(c, f"mkdir -p {REMOTE_SDK_DIR}/cmdline-tools")
    run(c, f"wget -q {ANDROID_CMDTOOLS_URL} -O /tmp/cmdtools.zip", timeout=120)
    run(c, f"unzip -q /tmp/cmdtools.zip -d /tmp/cmdtools_extracted")
    run(c, (
        f"mkdir -p {REMOTE_SDK_DIR}/cmdline-tools/latest && "
        f"mv /tmp/cmdtools_extracted/cmdline-tools/* {REMOTE_SDK_DIR}/cmdline-tools/latest/"
    ))

    # 환경변수 설정
    env_lines = (
        f"export ANDROID_SDK_ROOT={REMOTE_SDK_DIR}\\n"
        f"export PATH=$PATH:{REMOTE_SDK_DIR}/cmdline-tools/latest/bin"
        f":{REMOTE_SDK_DIR}/platform-tools"
        f":{REMOTE_SDK_DIR}/emulator"
    )
    run(c, f"grep -q ANDROID_SDK_ROOT ~/.bashrc || printf '{env_lines}' >> ~/.bashrc")

    sdkmanager = f"{REMOTE_SDK_DIR}/cmdline-tools/latest/bin/sdkmanager"
    # 라이선스 동의
    run(c, f"yes | ANDROID_SDK_ROOT={REMOTE_SDK_DIR} {sdkmanager} --licenses", timeout=30)
    # 필요 패키지 설치
    pkgs = f"'emulator' 'platform-tools' 'platforms;android-{ANDROID_API}' '{SYSTEM_IMAGE}'"
    run(c, f"ANDROID_SDK_ROOT={REMOTE_SDK_DIR} {sdkmanager} {pkgs}", timeout=300)
    print("  ✓ Android SDK 설치 완료")


def create_avd(c):
    step("4. AVD(에뮬레이터) 생성")
    avdmanager = f"{REMOTE_SDK_DIR}/cmdline-tools/latest/bin/avdmanager"
    existing = run(c, f"ANDROID_SDK_ROOT={REMOTE_SDK_DIR} {avdmanager} list avd 2>/dev/null", show=False)
    if AVD_NAME in existing:
        print(f"  ✓ AVD '{AVD_NAME}' 이미 존재, 스킵")
        return

    run(c, (
        f"echo 'no' | ANDROID_SDK_ROOT={REMOTE_SDK_DIR} {avdmanager} "
        f"create avd -n {AVD_NAME} -k '{SYSTEM_IMAGE}' --device '{AVD_DEVICE}' --force"
    ), timeout=60)
    print(f"  ✓ AVD '{AVD_NAME}' 생성 완료")


def setup_python_env(c):
    step("5. Python 가상환경 + 패키지 설치")
    run(c, f"python3 -m venv {REMOTE_PROJECT_DIR}/.venv", timeout=30)
    pip = f"{REMOTE_PROJECT_DIR}/.venv/bin/pip"
    run(c, f"{pip} install --upgrade pip -q", timeout=60)
    run(c, (
        f"{pip} install opencv-python-headless numpy paramiko -q"
    ), timeout=120)
    print("  ✓ Python 환경 준비 완료")


def create_start_script(c):
    step("6. 시작 스크립트 생성")
    run(c, f"mkdir -p {REMOTE_PROJECT_DIR}")

    # Xvfb + VNC + 에뮬레이터를 한 번에 시작하는 스크립트
    script = (
        "#!/bin/bash\\n"
        "set -e\\n"
        "export ANDROID_SDK_ROOT=" + REMOTE_SDK_DIR + "\\n"
        "# Xvfb 시작\\n"
        "Xvfb " + VNC_DISPLAY + " -screen 0 " + VNC_GEOMETRY + "x24 &\\n"
        "sleep 1\\n"
        "# VNC 서버 시작\\n"
        "vncserver " + VNC_DISPLAY + " -geometry " + VNC_GEOMETRY + " -depth 24 -localhost no\\n"
        "sleep 1\\n"
        "# 에뮬레이터 시작 (백그라운드)\\n"
        "DISPLAY=" + VNC_DISPLAY + " $ANDROID_SDK_ROOT/emulator/emulator "
        "-avd " + AVD_NAME + " -no-audio -gpu swiftshader_indirect "
        "-no-snapshot-save &\\n"
        "echo 'Waiting for emulator...'\\n"
        "adb wait-for-device\\n"
        "echo 'Emulator ready!'\\n"
    )
    start_path = f"{REMOTE_PROJECT_DIR}/start_env.sh"
    run(c, f"printf '{script}' > {start_path} && chmod +x {start_path}")

    # 종료 스크립트
    stop_script = (
        "#!/bin/bash\\n"
        "adb emu kill 2>/dev/null || true\\n"
        "vncserver -kill " + VNC_DISPLAY + " 2>/dev/null || true\\n"
        "pkill Xvfb 2>/dev/null || true\\n"
        "echo 'Stopped.'\\n"
    )
    stop_path = f"{REMOTE_PROJECT_DIR}/stop_env.sh"
    run(c, f"printf '{stop_script}' > {stop_path} && chmod +x {stop_path}")
    print(f"  ✓ {start_path}")
    print(f"  ✓ {stop_path}")


def main():
    print("=" * 55)
    print("  VLA Pipeline 서버 세팅")
    print(f"  {SERVER_HOST}:{SERVER_PORT}")
    print("=" * 55)

    c = connect()
    print("  ✓ SSH 연결 성공")

    setup_system(c)
    setup_vnc(c)
    setup_android_sdk(c)
    create_avd(c)
    create_start_script(c)
    setup_python_env(c)

    c.close()

    print("\n" + "=" * 55)
    print("  ✓ 세팅 완료!")
    print()
    print("  다음 단계:")
    print(f"  1. python deploy.py            ← 파이프라인 코드 업로드")
    print(f"  2. SSH 접속 후 실행:")
    print(f"       cd {REMOTE_PROJECT_DIR}")
    print(f"       bash start_env.sh")
    print(f"  3. RealVNC Viewer 접속: {SERVER_HOST}:5901")
    print(f"     비밀번호: {VNC_PASSWORD}")
    print("=" * 55)


if __name__ == "__main__":
    main()
