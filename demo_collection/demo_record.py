"""Demo recorder launcher — run this on your PC with the phone plugged in via USB.

Workflow:
  1) plug phone, enable USB debugging, accept RSA prompt
  2) python demo_record.py            # starts recording
  3) play your game on the phone
  4) python demo_stop.py              # graceful stop (writes meta.json)
  5) python demo_postprocess.py demos/<run_id>

Optional:
  python demo_record.py 60            # override max-fps (default 30)
  python demo_record.py --device <serial>
  python demo_record.py --out /path/to/demos

Environment overrides:
  ADB_PATH     full path to adb
  SCRCPY_PATH  full path to scrcpy
"""
import os, sys, subprocess, shutil, argparse

HERE = os.path.dirname(os.path.abspath(__file__))


def find_adb():
    if "ADB_PATH" in os.environ:
        return os.environ["ADB_PATH"]
    for n in ("adb.exe", "adb"):
        f = shutil.which(n)
        if f: return f
    for c in [
        os.path.expanduser("~/Android/Sdk/platform-tools/adb.exe"),
        os.path.expanduser("~/AppData/Local/Android/Sdk/platform-tools/adb.exe"),
        os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
        "/usr/bin/adb", "/usr/local/bin/adb",
        r"C:\platform-tools\adb.exe",
    ]:
        if os.path.exists(c): return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("max_fps", nargs="?", type=int, default=30,
                    help="scrcpy --max-fps (default 30)")
    ap.add_argument("--device", default=None, help="adb device serial")
    ap.add_argument("--out",    default=os.path.join(HERE, "demos"),
                    help="output root (default ./demos)")
    args = ap.parse_args()

    adb = find_adb()
    if not adb:
        print("[err] adb not found. Install Android platform-tools and put adb on PATH,")
        print("      or set the ADB_PATH environment variable.")
        sys.exit(2)
    print(f"[info] adb: {adb}")

    out = subprocess.check_output([adb, "devices"], text=True)
    print(out.strip())
    lines = [l for l in out.splitlines() if "\t" in l and "device" in l.split("\t")[1]]
    if not lines:
        print("[err] No device detected. Plug phone via USB + enable USB debugging.")
        sys.exit(2)
    device = args.device or lines[0].split("\t")[0]
    print(f"[info] using device: {device}")

    os.makedirs(args.out, exist_ok=True)

    env = dict(os.environ)
    env["ADB_PATH"] = adb
    env["PYTHONIOENCODING"] = "utf-8"

    rec = os.path.join(HERE, "demo_recorder.py")
    cmd = [sys.executable, "-u", rec,
           "--device",  device,
           "--max-fps", str(args.max_fps),
           "--out",     args.out]
    print(f"[info] launching: {' '.join(cmd)}")
    print(f"[info] output dir: {args.out}")
    print("[info] To stop: run `python demo_stop.py` (or Ctrl+C here).\n")

    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
