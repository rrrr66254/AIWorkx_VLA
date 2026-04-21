"""
로컬 vla_pipeline/ 코드를 서버에 SFTP로 업로드.
코드 수정할 때마다 실행.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import paramiko
from config import SERVER_HOST, SERVER_PORT, SERVER_USER, SERVER_PASS, REMOTE_PROJECT_DIR

LOCAL_DIR = os.path.join(os.path.dirname(__file__), "vla_pipeline")


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

    files = [f for f in os.listdir(LOCAL_DIR) if f.endswith(".py") or f.endswith(".txt")]
    print(f"업로드 대상: {len(files)}개 파일")

    for fname in files:
        local_path  = os.path.join(LOCAL_DIR, fname)
        remote_path = f"{REMOTE_PROJECT_DIR}/{fname}"
        sftp.put(local_path, remote_path)
        print(f"  ✓ {fname}")

    sftp.close()
    c.close()
    print(f"\n✓ 배포 완료 → {REMOTE_PROJECT_DIR}")


if __name__ == "__main__":
    deploy()
