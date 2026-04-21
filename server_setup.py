"""
One-time server setup script.
Run from local PC -> connects to server via paramiko -> auto-installs required packages/environment.
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
        print(f"  -> {stdout[:300]}")
    if stderr and "warning" not in stderr.lower():
        print(f"  [stderr] {stderr[:200]}")
    return stdout


def step(msg):
    print(f"\n{'─'*55}\n> {msg}\n{'─'*55}")


def setup_system(c):
    step("1. Install system packages")
    pkgs = (
        "xvfb tigervnc-standalone-server tigervnc-common "
        "adb python3-pip python3-venv "
        "openjdk-17-jdk-headless unzip wget curl libgl1"
    )
    run(c, f"sudo DEBIAN_FRONTEND=noninteractive apt-get install -y {pkgs}", timeout=180)
    print("  Package installation complete")


def setup_vnc(c):
    step("2. Configure VNC")
    run(c, "mkdir -p ~/.vnc")
    # Set password (non-interactive)
    run(c, f"printf '{VNC_PASSWORD}\\n{VNC_PASSWORD}\\nn\\n' | vncpasswd", timeout=10)
    # Create xstartup file
    xstartup = "#!/bin/sh\\nexec openbox-session &"
    run(c, f"echo '{xstartup}' > ~/.vnc/xstartup && chmod +x ~/.vnc/xstartup", timeout=5)
    # Install openbox (lightweight WM)
    run(c, "sudo apt-get install -y openbox", timeout=60)
    print("  VNC configuration complete")


def setup_android_sdk(c):
    step("3. Install Android SDK")
    # Skip if already installed
    if "cmdline-tools" in run(c, f"ls {REMOTE_SDK_DIR}/cmdline-tools 2>/dev/null || echo ''", show=False):
        print("  Already installed, skipping")
        return

    run(c, f"mkdir -p {REMOTE_SDK_DIR}/cmdline-tools")
    run(c, f"wget -q {ANDROID_CMDTOOLS_URL} -O /tmp/cmdtools.zip", timeout=120)
    run(c, f"unzip -q /tmp/cmdtools.zip -d /tmp/cmdtools_extracted")
    run(c, (
        f"mkdir -p {REMOTE_SDK_DIR}/cmdline-tools/latest && "
        f"mv /tmp/cmdtools_extracted/cmdline-tools/* {REMOTE_SDK_DIR}/cmdline-tools/latest/"
    ))

    # Set environment variables
    env_lines = (
        f"export ANDROID_SDK_ROOT={REMOTE_SDK_DIR}\\n"
        f"export PATH=$PATH:{REMOTE_SDK_DIR}/cmdline-tools/latest/bin"
        f":{REMOTE_SDK_DIR}/platform-tools"
        f":{REMOTE_SDK_DIR}/emulator"
    )
    run(c, f"grep -q ANDROID_SDK_ROOT ~/.bashrc || printf '{env_lines}' >> ~/.bashrc")

    sdkmanager = f"{REMOTE_SDK_DIR}/cmdline-tools/latest/bin/sdkmanager"
    # Accept licenses
    run(c, f"yes | ANDROID_SDK_ROOT={REMOTE_SDK_DIR} {sdkmanager} --licenses", timeout=30)
    # Install required packages
    pkgs = f"'emulator' 'platform-tools' 'platforms;android-{ANDROID_API}' '{SYSTEM_IMAGE}'"
    run(c, f"ANDROID_SDK_ROOT={REMOTE_SDK_DIR} {sdkmanager} {pkgs}", timeout=300)
    print("  Android SDK installation complete")


def create_avd(c):
    step("4. Create AVD (emulator)")
    avdmanager = f"{REMOTE_SDK_DIR}/cmdline-tools/latest/bin/avdmanager"
    existing = run(c, f"ANDROID_SDK_ROOT={REMOTE_SDK_DIR} {avdmanager} list avd 2>/dev/null", show=False)
    if AVD_NAME in existing:
        print(f"  AVD '{AVD_NAME}' already exists, skipping")
        return

    run(c, (
        f"echo 'no' | ANDROID_SDK_ROOT={REMOTE_SDK_DIR} {avdmanager} "
        f"create avd -n {AVD_NAME} -k '{SYSTEM_IMAGE}' --device '{AVD_DEVICE}' --force"
    ), timeout=60)
    print(f"  AVD '{AVD_NAME}' created successfully")


def setup_python_env(c):
    step("5. Python virtual environment + package installation")
    run(c, f"python3 -m venv {REMOTE_PROJECT_DIR}/.venv", timeout=30)
    pip = f"{REMOTE_PROJECT_DIR}/.venv/bin/pip"
    run(c, f"{pip} install --upgrade pip -q", timeout=60)
    run(c, (
        f"{pip} install opencv-python-headless numpy paramiko -q"
    ), timeout=120)
    print("  Python environment ready")


def create_start_script(c):
    step("6. Create startup scripts")
    run(c, f"mkdir -p {REMOTE_PROJECT_DIR}")

    # Script to start Xvfb + VNC + emulator all at once
    script = (
        "#!/bin/bash\\n"
        "set -e\\n"
        "export ANDROID_SDK_ROOT=" + REMOTE_SDK_DIR + "\\n"
        "# Start Xvfb\\n"
        "Xvfb " + VNC_DISPLAY + " -screen 0 " + VNC_GEOMETRY + "x24 &\\n"
        "sleep 1\\n"
        "# Start VNC server\\n"
        "vncserver " + VNC_DISPLAY + " -geometry " + VNC_GEOMETRY + " -depth 24 -localhost no\\n"
        "sleep 1\\n"
        "# Start emulator (background)\\n"
        "DISPLAY=" + VNC_DISPLAY + " $ANDROID_SDK_ROOT/emulator/emulator "
        "-avd " + AVD_NAME + " -no-audio -gpu swiftshader_indirect "
        "-no-snapshot-save &\\n"
        "echo 'Waiting for emulator...'\\n"
        "adb wait-for-device\\n"
        "echo 'Emulator ready!'\\n"
    )
    start_path = f"{REMOTE_PROJECT_DIR}/start_env.sh"
    run(c, f"printf '{script}' > {start_path} && chmod +x {start_path}")

    # Stop script
    stop_script = (
        "#!/bin/bash\\n"
        "adb emu kill 2>/dev/null || true\\n"
        "vncserver -kill " + VNC_DISPLAY + " 2>/dev/null || true\\n"
        "pkill Xvfb 2>/dev/null || true\\n"
        "echo 'Stopped.'\\n"
    )
    stop_path = f"{REMOTE_PROJECT_DIR}/stop_env.sh"
    run(c, f"printf '{stop_script}' > {stop_path} && chmod +x {stop_path}")
    print(f"  {start_path}")
    print(f"  {stop_path}")


def main():
    print("=" * 55)
    print("  VLA Pipeline Server Setup")
    print(f"  {SERVER_HOST}:{SERVER_PORT}")
    print("=" * 55)

    c = connect()
    print("  SSH connection successful")

    setup_system(c)
    setup_vnc(c)
    setup_android_sdk(c)
    create_avd(c)
    create_start_script(c)
    setup_python_env(c)

    c.close()

    print("\n" + "=" * 55)
    print("  Setup complete!")
    print()
    print("  Next steps:")
    print(f"  1. python deploy.py            <- upload pipeline code")
    print(f"  2. Connect via SSH and run:")
    print(f"       cd {REMOTE_PROJECT_DIR}")
    print(f"       bash start_env.sh")
    print(f"  3. Connect with RealVNC Viewer: {SERVER_HOST}:5901")
    print(f"     Password: {VNC_PASSWORD}")
    print("=" * 55)


if __name__ == "__main__":
    main()
