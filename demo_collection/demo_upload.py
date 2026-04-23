"""(Optional) Upload a local demo run to NAS (raw) + trigger remote postprocess.

Only needed if you have a shared NAS + a GPU server and want to keep raw mp4s
off your local disk. For local-only workflows, skip this and just run
`demo_postprocess.py` directly.

Setup:
  cp config.example.py config.py
  # edit SERVER_HOST / SERVER_USER / SERVER_PASS / NAS_BASE / SERVER_BASE

Usage:
  python demo_upload.py <run_id>
  python demo_upload.py <run_id> --no-process     # upload to NAS only
  python demo_upload.py <run_id> --delete-local   # delete local mp4 after upload
"""
import sys, io, os, time

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import paramiko
except ImportError:
    print("[err] paramiko not installed. pip install paramiko"); sys.exit(2)

try:
    from config import (SERVER_HOST, SERVER_PORT, SERVER_USER, SERVER_PASS,
                        NAS_BASE, SERVER_BASE, REMOTE_PYTHON, REMOTE_POSTPROCESS,
                        FRAME_STRIDE, FRAME_SIZE, JPG_QUALITY)
except ImportError:
    print("[err] config.py not found.")
    print("      cp config.example.py config.py  and fill in your server info.")
    sys.exit(2)


def sftp_mkdirs(sftp, path):
    parts = []
    while path and path != "/":
        parts.append(path); path = os.path.dirname(path)
    for p in reversed(parts):
        try: sftp.stat(p)
        except IOError:
            try: sftp.mkdir(p)
            except IOError: pass


def put_with_progress(sftp, local, remote, label=""):
    sz = os.path.getsize(local); t0 = time.time(); last = [0, t0]
    def cb(done, total):
        now = time.time()
        if now - last[1] >= 2.0 or done == total:
            mb, tot = done/1e6, total/1e6
            rate = (done-last[0]) / max(now-last[1], 1e-3) / 1e6
            print(f"  [{label}] {mb:.1f}/{tot:.1f} MB  {100.0*done/max(total,1):.1f}%  {rate:.1f} MB/s")
            last[0] = done; last[1] = now
    sftp.put(local, remote, callback=cb)
    dt = time.time()-t0
    print(f"  [{label}] done {sz/1e6:.1f} MB in {dt:.1f}s ({sz/1e6/max(dt,1e-3):.1f} MB/s)")


def main():
    if len(sys.argv) < 2:
        print("usage: python demo_upload.py <run_id> [--no-process] [--delete-local]"); sys.exit(2)
    run_id       = sys.argv[1]
    do_process   = "--no-process"   not in sys.argv
    delete_local = "--delete-local" in sys.argv

    local_dir = os.path.join(HERE, "demos", run_id)
    if not os.path.isdir(local_dir):
        print(f"[err] not found: {local_dir}"); sys.exit(2)

    raw_files = ["video.mp4", "events.jsonl", "meta.json", "recorder.log"]
    total_sz = sum(os.path.getsize(os.path.join(local_dir, f))
                   for f in raw_files if os.path.exists(os.path.join(local_dir, f)))
    print(f"[info] local {local_dir}: raw size = {total_sz/1e6:.1f} MB")

    nas_dir    = f"{NAS_BASE}/{run_id}"
    server_dir = f"{SERVER_BASE}/{run_id}"
    print(f"[info] NAS target:    {nas_dir}")
    print(f"[info] server target: {server_dir}")

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(SERVER_HOST, SERVER_PORT, SERVER_USER, SERVER_PASS, timeout=15)
    sftp = c.open_sftp()

    sftp_mkdirs(sftp, nas_dir)
    t0 = time.time()
    for fn in raw_files:
        lp = os.path.join(local_dir, fn)
        if not os.path.exists(lp): continue
        put_with_progress(sftp, lp, f"{nas_dir}/{fn}", label=fn)
    print(f"[up] NAS upload elapsed {time.time()-t0:.1f}s")

    # sync postprocess script (keeps server in sync with local version)
    sftp.put(os.path.join(HERE, "demo_postprocess.py"), REMOTE_POSTPROCESS)
    sftp.close()

    if do_process:
        print(f"[proc] purging stale server outputs ...")
        _, o, _ = c.exec_command(f"rm -rf {server_dir}/frames {server_dir}/video.mp4 "
                                 f"{server_dir}/events.jsonl && echo OK")
        print(o.read().decode().strip())
        cmd = (f"{REMOTE_PYTHON} -u {REMOTE_POSTPROCESS} {nas_dir} "
               f"--out-dir {server_dir} "
               f"--frame-stride {FRAME_STRIDE} "
               f"--frame-size {FRAME_SIZE} "
               f"--jpg-quality {JPG_QUALITY}")
        print(f"[proc] {cmd}")
        _, o, e = c.exec_command(cmd)
        for ln in iter(o.readline, ""):
            sys.stdout.write(ln); sys.stdout.flush()
        err = e.read().decode()
        if err.strip(): print("[stderr]\n" + err)

        _, o, _ = c.exec_command(f"cat {server_dir}/summary.json 2>/dev/null")
        print("[summary]\n" + o.read().decode())
        _, o, _ = c.exec_command(f"du -sh {server_dir} {nas_dir} 2>/dev/null")
        print("[du]\n" + o.read().decode())

    c.close()

    if delete_local:
        mp4 = os.path.join(local_dir, "video.mp4")
        if os.path.exists(mp4):
            sz = os.path.getsize(mp4); os.remove(mp4)
            print(f"[cleanup] deleted local video.mp4 ({sz/1e6:.1f} MB freed)")


if __name__ == "__main__":
    main()
