"""Config template for demo_upload.py. Copy to config.py and fill in.

  cp config.example.py config.py

config.py is git-ignored.
Only needed if you use the optional remote-upload workflow.
"""

# --- SSH target (your GPU server) ---
SERVER_HOST = "your.server.ip.or.host"
SERVER_PORT = 22
SERVER_USER = "your_user"
SERVER_PASS = "your_password"       # consider key-based auth for production

# --- Remote paths ---
# NAS mount on the server where raw mp4 / events / meta are stored
NAS_BASE    = "/mnt/nas/your_org/demos_raw"
# Local (non-NAS) disk on the server where postprocess writes downsampled frames
SERVER_BASE = "/home/your_user/demos"

# Remote Python + postprocess script path
REMOTE_PYTHON      = "/usr/bin/python3"
REMOTE_POSTPROCESS = "/home/your_user/demo_postprocess.py"

# --- Postprocess settings (applied on the server) ---
FRAME_STRIDE = 3     # 30fps -> 10fps
FRAME_SIZE   = 384   # longest edge px (0 = native)
JPG_QUALITY  = 88
