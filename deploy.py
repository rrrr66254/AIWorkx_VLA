"""
Upload local vla_pipeline/ code to server via SFTP.
Run this every time code is modified.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import paramiko
from config import SERVER_HOST, SERVER_PORT, SERVER_USER, SERVER_PASS, REMOTE_PROJECT_DIR

ROOT_DIR  = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.join(ROOT_DIR, "vla_pipeline")

# Root-level files that also run on the server
ROOT_FILES = [
    "pipeline_ng_rl.py",
    "fast_capture.py",
    "rl_agent.py",
]


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(SERVER_HOST, SERVER_PORT, SERVER_USER, SERVER_PASS, timeout=15)
    return c


def run(c, cmd):
    _, out, _ = c.exec_command(cmd, timeout=15)
    return out.read().decode("utf-8", errors="ignore").strip()


def deploy():
    c = connect()
    sftp = c.open_sftp()

    run(c, f"mkdir -p {REMOTE_PROJECT_DIR}")

    # 1) vla_pipeline/ module files
    module_files = [f for f in os.listdir(LOCAL_DIR) if f.endswith(".py") or f.endswith(".txt")]
    all_uploads = [(os.path.join(LOCAL_DIR, f), f) for f in module_files]

    # 2) Root-level server files (pipeline, rl_agent, fast_capture)
    for fname in ROOT_FILES:
        local_path = os.path.join(ROOT_DIR, fname)
        if os.path.exists(local_path):
            all_uploads.append((local_path, fname))

    print(f"Upload targets: {len(all_uploads)} files")
    for local_path, fname in all_uploads:
        remote_path = f"{REMOTE_PROJECT_DIR}/{fname}"
        sftp.put(local_path, remote_path)
        print(f"  {fname}")

    sftp.close()
    c.close()
    print(f"\nDeploy complete -> {REMOTE_PROJECT_DIR}")


if __name__ == "__main__":
    deploy()
