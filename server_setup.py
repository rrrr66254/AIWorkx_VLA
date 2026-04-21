"""
One-time server setup script.

What this does:
  1. Install system packages  (Xvfb, ADB, OpenJDK, git, ...)
  2. Install Miniconda         (if not already present)
  3. Install pip packages      (torch, opencv, scrcpy-client, zmq, ...)
  4. Install Android SDK       (cmdline-tools + platform-tools + API 34 emulator image)
  5. Create AVD                (GameTest / Pixel 4)
  6. Clone + install NitroGen  (skip if already present)
  7. Deploy pipeline code      (vla_pipeline/ -> ~/vla_pipeline/)
  8. Create helper scripts     (start_emulator.sh / stop_emulator.sh)

Monitoring is done entirely via viewer.py (SSH + ADB screencap / scrcpy).
No VNC is installed or needed.

Run from local PC:
  python server_setup.py
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import paramiko
from config import (
    SERVER_HOST, SERVER_PORT, SERVER_USER, SERVER_PASS,
    REMOTE_SDK_DIR, REMOTE_PROJECT_DIR, REMOTE_HOME,
    NITROGEN_DIR, PYTHON_PATH,
    AVD_NAME, AVD_DEVICE, ANDROID_API, SYSTEM_IMAGE,
)

ANDROID_CMDTOOLS_URL = (
    "https://dl.google.com/android/repository/"
    "commandlinetools-linux-11076708_latest.zip"
)
MINICONDA_URL = (
    "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
)

LOCAL_PIPELINE_DIR = os.path.join(os.path.dirname(__file__), "vla_pipeline")


# ── Helpers ──────────────────────────────────────────────────────────────────

def connect() -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(SERVER_HOST, SERVER_PORT, SERVER_USER, SERVER_PASS, timeout=15)
    return c


def run(c, cmd, timeout=120, show=True) -> str:
    _, out, err = c.exec_command(cmd, timeout=timeout)
    stdout = out.read().decode("utf-8", errors="ignore").strip()
    stderr = err.read().decode("utf-8", errors="ignore").strip()
    if show and stdout:
        print(f"    {stdout[:400]}")
    if stderr and "warning" not in stderr.lower() and show:
        print(f"    [stderr] {stderr[:200]}")
    return stdout


def step(n, msg):
    print(f"\n{'─'*60}\n  Step {n}: {msg}\n{'─'*60}")


# ── Setup Steps ───────────────────────────────────────────────────────────────

def setup_system(c):
    step(1, "Install system packages")
    pkgs = (
        "xvfb "                        # virtual framebuffer (headless display for emulator)
        "adb "                         # Android Debug Bridge
        "openjdk-17-jdk-headless "     # required by Android SDK tools
        "python3-pip python3-venv "
        "unzip wget curl git "
        "libgl1 libpulse0 "            # emulator runtime dependencies
        "libxcb-xinerama0 libxcb1 "
    )
    run(c, f"sudo DEBIAN_FRONTEND=noninteractive apt-get install -y {pkgs}", timeout=180)
    print("  System packages installed.")


def setup_miniconda(c):
    step(2, "Install Miniconda")
    already = run(c, f"test -f {PYTHON_PATH} && echo yes || echo no", show=False)
    if already.strip() == "yes":
        print("  Miniconda already present, skipping.")
        return
    run(c, f"wget -q {MINICONDA_URL} -O /tmp/miniconda.sh", timeout=120)
    run(c, "bash /tmp/miniconda.sh -b -p ~/miniconda3", timeout=180)
    run(c, "~/miniconda3/bin/conda init bash", timeout=30)
    print("  Miniconda installed.")


def setup_pip_packages(c):
    step(3, "Install pip packages")
    pip = PYTHON_PATH.replace("python3", "pip") if "python3" in PYTHON_PATH else f"{os.path.dirname(PYTHON_PATH)}/pip"
    pkgs = (
        "torch torchvision --index-url https://download.pytorch.org/whl/cu121 "
    )
    run(c, f"{pip} install --upgrade pip -q", timeout=60)
    # torch (CUDA 12.1 build — matches CUDA 12.x)
    run(c, f"{pip} install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q", timeout=300)
    # other dependencies
    run(c, f"{pip} install opencv-python-headless numpy paramiko scrcpy-client pyzmq -q", timeout=120)
    print("  Pip packages installed.")


def setup_android_sdk(c):
    step(4, "Install Android SDK")
    check = run(c, f"ls {REMOTE_SDK_DIR}/cmdline-tools 2>/dev/null || echo ''", show=False)
    if "latest" in check:
        print("  Android SDK already installed, skipping.")
    else:
        run(c, f"mkdir -p {REMOTE_SDK_DIR}/cmdline-tools")
        run(c, f"wget -q {ANDROID_CMDTOOLS_URL} -O /tmp/cmdtools.zip", timeout=120)
        run(c, "unzip -q /tmp/cmdtools.zip -d /tmp/cmdtools_extracted")
        run(c, (
            f"mkdir -p {REMOTE_SDK_DIR}/cmdline-tools/latest && "
            f"mv /tmp/cmdtools_extracted/cmdline-tools/* {REMOTE_SDK_DIR}/cmdline-tools/latest/"
        ))
        # Persist environment variables
        env_block = (
            f"export ANDROID_SDK_ROOT={REMOTE_SDK_DIR}\\n"
            f"export PATH=$PATH:{REMOTE_SDK_DIR}/cmdline-tools/latest/bin"
            f":{REMOTE_SDK_DIR}/platform-tools"
            f":{REMOTE_SDK_DIR}/emulator"
        )
        run(c, f"grep -q ANDROID_SDK_ROOT ~/.bashrc || printf '{env_block}\\n' >> ~/.bashrc")
        print("  Android SDK downloaded.")

    sdkmanager = f"{REMOTE_SDK_DIR}/cmdline-tools/latest/bin/sdkmanager"
    run(c, f"yes | ANDROID_SDK_ROOT={REMOTE_SDK_DIR} {sdkmanager} --licenses 2>/dev/null", timeout=30)
    pkgs = f"'emulator' 'platform-tools' 'platforms;android-{ANDROID_API}' '{SYSTEM_IMAGE}'"
    run(c, f"ANDROID_SDK_ROOT={REMOTE_SDK_DIR} {sdkmanager} {pkgs}", timeout=300)
    print("  Android SDK packages ready.")


def create_avd(c):
    step(5, "Create AVD")
    avdmanager = f"{REMOTE_SDK_DIR}/cmdline-tools/latest/bin/avdmanager"
    existing = run(c, f"ANDROID_SDK_ROOT={REMOTE_SDK_DIR} {avdmanager} list avd 2>/dev/null", show=False)
    if AVD_NAME in existing:
        print(f"  AVD '{AVD_NAME}' already exists, skipping.")
        return
    run(c, (
        f"echo 'no' | ANDROID_SDK_ROOT={REMOTE_SDK_DIR} {avdmanager} "
        f"create avd -n {AVD_NAME} -k '{SYSTEM_IMAGE}' --device '{AVD_DEVICE}' --force"
    ), timeout=60)
    print(f"  AVD '{AVD_NAME}' created.")


def setup_nitrogen(c):
    step(6, "Clone + install NitroGen")
    already = run(c, f"test -d {NITROGEN_DIR} && echo yes || echo no", show=False)
    if already.strip() == "yes":
        print("  NitroGen directory already exists, skipping clone.")
    else:
        run(c, f"git clone https://github.com/MineDojo/NitroGen {NITROGEN_DIR}", timeout=120)
        print("  NitroGen cloned.")

    pip = PYTHON_PATH.replace("python3", "pip") if "python3" in PYTHON_PATH else f"{os.path.dirname(PYTHON_PATH)}/pip"
    run(c, f"cd {NITROGEN_DIR} && {pip} install -e '.[serve]' -q", timeout=180)
    print("  NitroGen installed.")
    print(f"  Place model weights at: {NITROGEN_DIR}/ng.pt")


def deploy_pipeline(c):
    step(7, "Deploy pipeline code")
    run(c, f"mkdir -p {REMOTE_PROJECT_DIR}")
    sftp = c.open_sftp()
    files = [f for f in os.listdir(LOCAL_PIPELINE_DIR) if f.endswith(".py") or f.endswith(".txt")]
    for fname in files:
        sftp.put(os.path.join(LOCAL_PIPELINE_DIR, fname), f"{REMOTE_PROJECT_DIR}/{fname}")
        print(f"  -> {fname}")
    sftp.close()
    print(f"  Pipeline code deployed to {REMOTE_PROJECT_DIR}")


def create_scripts(c):
    step(8, "Create helper scripts")
    run(c, f"mkdir -p {REMOTE_PROJECT_DIR}")

    adb = f"{REMOTE_SDK_DIR}/platform-tools/adb"
    emulator = f"{REMOTE_SDK_DIR}/emulator/emulator"

    # start_emulator.sh ─────────────────────────────────────────────
    # Starts Xvfb on :1, then launches the Android emulator on that display.
    # The pipeline and viewer.py connect via ADB over TCP (no VNC needed).
    start = (
        "#!/bin/bash\\n"
        "set -e\\n"
        f"export ANDROID_SDK_ROOT={REMOTE_SDK_DIR}\\n"
        "\\n"
        "# Kill stale processes\\n"
        "pkill Xvfb 2>/dev/null; sleep 0.5\\n"
        f"{adb} kill-server 2>/dev/null; sleep 0.3\\n"
        f"{adb} start-server\\n"
        "\\n"
        "# Virtual framebuffer (headless display)\\n"
        "Xvfb :1 -screen 0 1280x800x24 &\\n"
        "sleep 2\\n"
        "\\n"
        "# Android emulator\\n"
        f"DISPLAY=:1 {emulator} \\\\\\n"
        f"  -avd {AVD_NAME} \\\\\\n"
        "  -no-audio -gpu swiftshader_indirect \\\\\\n"
        "  -no-boot-anim -no-snapshot-save &\\n"
        "\\n"
        "echo 'Waiting for emulator to boot...'\\n"
        f"{adb} wait-for-device\\n"
        f"{adb} shell input keyevent 82   # unlock screen\\n"
        "echo 'Emulator ready.'\\n"
        "echo 'Connect via: python viewer.py  (http://localhost:8080)'\\n"
    )
    run(c, f"printf '{start}' > {REMOTE_PROJECT_DIR}/start_emulator.sh && "
           f"chmod +x {REMOTE_PROJECT_DIR}/start_emulator.sh")

    # stop_emulator.sh ──────────────────────────────────────────────
    stop = (
        "#!/bin/bash\\n"
        f"{adb} emu kill 2>/dev/null || true\\n"
        "pkill -f pipeline_ng_rl.py 2>/dev/null || true\\n"
        "pkill -f 'serve.py.*ng.pt' 2>/dev/null || true\\n"
        "pkill Xvfb 2>/dev/null || true\\n"
        "echo 'Stopped.'\\n"
    )
    run(c, f"printf '{stop}' > {REMOTE_PROJECT_DIR}/stop_emulator.sh && "
           f"chmod +x {REMOTE_PROJECT_DIR}/stop_emulator.sh")

    # watchdog_ng_rl.sh ─────────────────────────────────────────────
    watchdog = (
        "#!/bin/bash\\n"
        f"PYTHON='{PYTHON_PATH}'\\n"
        f"PIPELINE='{REMOTE_PROJECT_DIR}/pipeline_ng_rl.py'\\n"
        f"LOG='{REMOTE_PROJECT_DIR}/pipeline_ng_rl.log'\\n"
        "DEVICE='emulator-5554'\\n"
        "\\n"
        "echo '[watchdog] starting pipeline loop'\\n"
        "while true; do\\n"
        "    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 DISPLAY=:1 \\\\\\n"
        "        $PYTHON -u $PIPELINE \\\\\\n"
        "        --device $DEVICE \\\\\\n"
        "        --step-interval 0.1 \\\\\\n"
        "        --nitrogen-port 5556 \\\\\\n"
        "        --no-record >> $LOG 2>&1\\n"
        "    echo \"[watchdog] restart $(date)\" >> $LOG\\n"
        "    sleep 5\\n"
        "done\\n"
    )
    run(c, f"printf '{watchdog}' > {REMOTE_PROJECT_DIR}/watchdog_ng_rl.sh && "
           f"chmod +x {REMOTE_PROJECT_DIR}/watchdog_ng_rl.sh")

    print(f"  {REMOTE_PROJECT_DIR}/start_emulator.sh")
    print(f"  {REMOTE_PROJECT_DIR}/stop_emulator.sh")
    print(f"  {REMOTE_PROJECT_DIR}/watchdog_ng_rl.sh")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  VLA Pipeline Server Setup")
    print(f"  {SERVER_USER}@{SERVER_HOST}:{SERVER_PORT}")
    print("=" * 60)

    c = connect()
    print("  SSH connection successful.\n")

    setup_system(c)
    setup_miniconda(c)
    setup_pip_packages(c)
    setup_android_sdk(c)
    create_avd(c)
    setup_nitrogen(c)
    deploy_pipeline(c)
    create_scripts(c)

    c.close()

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print()
    print("  Next steps (run from server SSH):")
    print(f"    bash {REMOTE_PROJECT_DIR}/start_emulator.sh")
    print()
    print("  Install the game APK (first time only):")
    print(f"    {REMOTE_SDK_DIR}/platform-tools/adb install subway-surfers.apk")
    print()
    print("  Place NitroGen weights:")
    print(f"    ~/NitroGen/ng.pt")
    print()
    print("  Then from local PC:")
    print("    python viewer.py   ->   http://localhost:8080")
    print("    Click 'Start NitroGen' to begin training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
