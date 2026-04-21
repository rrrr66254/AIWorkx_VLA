"""
config.example.py  --  Template for personal server configuration.

Steps:
  1. Copy this file:   cp config.example.py config.py
  2. Fill in your own server credentials and username below.
  3. config.py is listed in .gitignore and will never be committed.

Alternatively, set environment variables instead of editing the file:
  export VLA_HOST=YOUR_SERVER_IP
  export VLA_USER=YOUR_USERNAME
  export VLA_PASS=YOUR_PASSWORD
  export VLA_VNC_PASS=YOUR_VNC_PASSWORD
"""
import os

# ── Server connection info ──────────────────────────────────────────────
SERVER_HOST = os.environ.get("VLA_HOST", "YOUR_SERVER_IP")   # e.g. "192.168.1.100"
SERVER_PORT = int(os.environ.get("VLA_PORT", "22"))
SERVER_USER = os.environ.get("VLA_USER", "YOUR_USERNAME")     # e.g. "john"
SERVER_PASS = os.environ.get("VLA_PASS", "YOUR_PASSWORD")

# ── VNC settings ────────────────────────────────────────────────────────
VNC_DISPLAY  = ":1"
VNC_PORT     = 5901
VNC_PASSWORD = os.environ.get("VLA_VNC_PASS", "YOUR_VNC_PASSWORD")
VNC_GEOMETRY = "1280x800"

# ── Remote paths (auto-derived from SERVER_USER) ────────────────────────
# These do not need to be changed unless you use a non-standard home directory.
REMOTE_HOME        = f"/home/{SERVER_USER}"
REMOTE_PROJECT_DIR = f"{REMOTE_HOME}/vla_pipeline"
REMOTE_SDK_DIR     = f"{REMOTE_HOME}/android-sdk"
REMOTE_OUTPUT_DIR  = f"{REMOTE_PROJECT_DIR}/output"

NITROGEN_DIR  = f"{REMOTE_HOME}/NitroGen"
NITROGEN_CKPT = f"{NITROGEN_DIR}/ng.pt"

# Python interpreter on the server.
# - miniconda (default): ~/miniconda3/bin/python3
# - system python:       /usr/bin/python3
PYTHON_PATH = f"{REMOTE_HOME}/miniconda3/bin/python3"

# ── Android emulator settings ────────────────────────────────────────────
AVD_NAME       = "GameTest"
AVD_DEVICE     = "pixel_4"
ANDROID_API    = "34"
SYSTEM_IMAGE   = f"system-images;android-{ANDROID_API};google_apis;x86_64"
EMULATOR_PORT  = 5554

# ── Pipeline settings ────────────────────────────────────────────────────
STEP_INTERVAL  = 0.5
CAPTURE_WIDTH  = 256
CAPTURE_HEIGHT = 256
