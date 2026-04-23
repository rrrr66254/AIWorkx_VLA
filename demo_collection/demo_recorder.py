"""Demo recorder v2 — scrcpy-based high-fps capture.

Captures:
  - video.mp4 via scrcpy (H.264, 30fps)
  - events.jsonl via `adb shell getevent -lt`

Graceful stop: create a file named "stop.flag" in the run directory, or send
SIGINT (Ctrl+C). SIGTERM is best-effort on Windows.

Output tree:
  demos/{run_id}/
    video.mp4
    events.jsonl
    meta.json
    recorder.log
    recorder.pid

Frame extraction happens in demo_postprocess.py (reads mp4 and writes
frame_ts.jsonl + actions.jsonl).
"""
from __future__ import annotations
import argparse, json, os, re, shutil, signal, subprocess, sys, threading, time
from pathlib import Path
from typing import Optional


def _find(paths, name):
    for p in paths:
        q = os.path.expanduser(p)
        if os.path.exists(q): return q
    return shutil.which(name) or name


def default_adb() -> str:
    if "ADB_PATH" in os.environ: return os.environ["ADB_PATH"]
    return _find([
        "~/android-sdk/platform-tools/adb",
        "~/Android/Sdk/platform-tools/adb.exe",
        "~/AppData/Local/Android/Sdk/platform-tools/adb.exe",
        r"C:\platform-tools\adb.exe",
    ], "adb")


def default_scrcpy() -> str:
    if "SCRCPY_PATH" in os.environ: return os.environ["SCRCPY_PATH"]
    return _find([
        "~/AppData/Local/Microsoft/WinGet/Links/scrcpy.exe",
        "/usr/bin/scrcpy",
        "/usr/local/bin/scrcpy",
        "/opt/homebrew/bin/scrcpy",
    ], "scrcpy")


ADB    = default_adb()
SCRCPY = default_scrcpy()


def ts_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


class EventReader(threading.Thread):
    """Spawns `adb getevent -lt` and writes each line to events.jsonl."""
    def __init__(self, device: str, out_path: Path):
        super().__init__(daemon=True)
        self.device = device
        self.out_path = out_path
        self.proc: Optional[subprocess.Popen] = None
        self.stop_flag = threading.Event()
        self.line_count = 0
        self.first_wall: Optional[float] = None

    def run(self):
        cmd = [ADB, "-s", self.device, "shell", "getevent", "-lt"]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=1, text=True, errors="ignore",
        )
        with open(self.out_path, "a", encoding="utf-8") as f:
            for line in self.proc.stdout:
                if self.stop_flag.is_set(): break
                now = time.time()
                if self.first_wall is None: self.first_wall = now
                rec = {"wall": now, "raw": line.rstrip("\n")}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self.line_count += 1

    def stop(self):
        self.stop_flag.set()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate(); self.proc.wait(timeout=3)
            except Exception:
                try: self.proc.kill()
                except Exception: pass


def get_screen_size(device: str) -> tuple[int, int]:
    out = subprocess.check_output([ADB, "-s", device, "shell", "wm", "size"],
                                  text=True, timeout=5)
    for tok in out.split():
        if "x" in tok and tok.replace("x","").isdigit():
            w, h = tok.split("x"); return int(w), int(h)
    raise RuntimeError(f"failed to parse screen size: {out!r}")


def detect_touch_abs_max(device: str) -> tuple[int, int, str]:
    try:
        out = subprocess.check_output([ADB, "-s", device, "shell", "getevent", "-pl"],
                                      text=True, timeout=10, errors="ignore")
    except Exception:
        return 32767, 32767, "unknown"
    blocks = []; cur = []
    for ln in out.splitlines():
        if ln.startswith("add device"):
            if cur: blocks.append(cur)
            cur = [ln]
        else: cur.append(ln)
    if cur: blocks.append(cur)
    candidates = []
    # PREFER the exact "touchscreen" keyword over "touchpad"
    for blk in blocks:
        text = "\n".join(blk)
        if "ABS_MT_POSITION_X" not in text: continue
        name = ""
        for ln in blk:
            m = re.search(r'name:\s*"([^"]+)"', ln)
            if m: name = m.group(1); break
        has_btn = "BTN_TOUCH" in text
        nl = name.lower()
        is_screen = ("touchscreen" in nl) or ("touch_screen" in nl) or \
                    ("input_multi_touch" in nl)
        is_touchpad = "touchpad" in nl
        mx = my = 0
        for ln in blk:
            m = re.search(r'ABS_MT_POSITION_X\s*:.*?max\s+(\d+)', ln)
            if m: mx = int(m.group(1))
            m = re.search(r'ABS_MT_POSITION_Y\s*:.*?max\s+(\d+)', ln)
            if m: my = int(m.group(1))
        score = 0
        if is_screen:    score += 200
        elif is_touchpad: score -= 50
        if has_btn:      score += 10
        score += mx / 10000.0
        candidates.append((score, name, mx, my))
    if not candidates: return 32767, 32767, "unknown"
    candidates.sort(key=lambda x: -x[0])
    _, name, mx, my = candidates[0]
    if mx <= 0: mx = 32767
    if my <= 0: my = 32767
    return mx, my, name


def spawn_scrcpy(device: str, mp4_path: Path, max_fps: int,
                 bit_rate: str, max_size: int, log_path: Path) -> subprocess.Popen:
    cmd = [
        SCRCPY, "-s", device,
        "--no-playback",
        "--no-control",
        "--no-audio",
        "--record", str(mp4_path),
        f"--max-fps={max_fps}",
        f"--video-bit-rate={bit_rate}",
    ]
    if max_size > 0:
        cmd.append(f"--max-size={max_size}")
    log_f = open(log_path, "w", encoding="utf-8")
    log_f.write(f"CMD: {' '.join(cmd)}\n"); log_f.flush()
    # scrcpy listens for SIGINT on stdin close / Ctrl+C; use Popen without new console so
    # we can stop it cleanly via terminate() on Windows (sends CTRL_BREAK to the group).
    kwargs = dict(stdout=log_f, stderr=subprocess.STDOUT)
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(cmd, **kwargs)
    return proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",   default="emulator-5554")
    ap.add_argument("--out",      default=os.path.expanduser("~/demos"))
    ap.add_argument("--run-id",   default=None)
    ap.add_argument("--max-fps",  type=int, default=30)
    ap.add_argument("--bit-rate", default="8M")
    ap.add_argument("--max-size", type=int, default=0,
                    help="downscale longest edge to N px (0 = keep native)")
    ap.add_argument("--max-seconds", type=float, default=0)
    args = ap.parse_args()

    run_id = args.run_id or ts_run_id()
    root   = Path(args.out) / run_id
    root.mkdir(parents=True, exist_ok=True)

    mp4_path    = root / "video.mp4"
    events_path = root / "events.jsonl"
    meta_path   = root / "meta.json"
    log_path    = root / "recorder.log"
    pid_path    = root / "recorder.pid"
    stop_flag   = root / "stop.flag"

    pid_path.write_text(str(os.getpid()))
    if stop_flag.exists(): stop_flag.unlink()

    w, h = get_screen_size(args.device)
    abs_mx, abs_my, touch_name = detect_touch_abs_max(args.device)
    print(f"[rec] device={args.device}  screen={w}x{h}  run_id={run_id}")
    print(f"[rec] touch='{touch_name}'  abs=({abs_mx},{abs_my})")
    print(f"[rec] scrcpy={SCRCPY}")
    print(f"[rec] out={root}")
    print(f"[rec] TO STOP: create file '{stop_flag}' OR Ctrl+C")

    # Start event reader
    ev = EventReader(args.device, events_path)
    ev.start()

    # Start scrcpy
    scrcpy_wall_start = time.time()
    sc = spawn_scrcpy(args.device, mp4_path, args.max_fps,
                      args.bit_rate, args.max_size, log_path)
    time.sleep(1.0)   # let scrcpy warm up
    if sc.poll() is not None:
        print("[rec] scrcpy died immediately. check recorder.log")
        ev.stop(); sys.exit(2)
    print(f"[rec] scrcpy PID={sc.pid} max_fps={args.max_fps} bit_rate={args.bit_rate}")

    running = True
    def _stop(_sig=None, _frm=None):
        nonlocal running
        running = False
        print("[rec] stop requested")
    signal.signal(signal.SIGINT, _stop)
    try: signal.signal(signal.SIGTERM, _stop)
    except (AttributeError, ValueError): pass

    t0 = time.time()
    try:
        while running:
            if stop_flag.exists():
                print("[rec] stop.flag detected")
                break
            if sc.poll() is not None:
                print(f"[rec] scrcpy exited unexpectedly (code={sc.returncode})")
                break
            if args.max_seconds and (time.time() - t0) >= args.max_seconds:
                print(f"[rec] max-seconds reached ({args.max_seconds}s)")
                break
            time.sleep(0.5)
    finally:
        end_wall = time.time()
        # 1) stop scrcpy gracefully
        if sc.poll() is None:
            print("[rec] terminating scrcpy...")
            try:
                if os.name == "nt":
                    # SIGBREAK lets scrcpy flush the mp4 trailer (vs TerminateProcess)
                    sc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    sc.terminate()
            except Exception as e:
                print(f"[rec] terminate err: {e}")
            try: sc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("[rec] scrcpy didn't exit in 10s, killing")
                sc.kill(); sc.wait()
        # 2) stop event reader
        ev.stop(); ev.join(timeout=5)
        # 3) meta.json
        meta = {
            "run_id":             run_id,
            "device":             args.device,
            "screen_w":           w,
            "screen_h":           h,
            "touch_name":         touch_name,
            "abs_max_x":          abs_mx,
            "abs_max_y":          abs_my,
            "abs_max":            max(abs_mx, abs_my),
            "target_fps":         args.max_fps,
            "bit_rate":           args.bit_rate,
            "max_size":           args.max_size,
            "scrcpy_wall_start":  scrcpy_wall_start,
            "scrcpy_wall_end":    end_wall,
            "duration_s":         end_wall - scrcpy_wall_start,
            "video_path":         "video.mp4",
            "video_size_bytes":   mp4_path.stat().st_size if mp4_path.exists() else 0,
            "event_count":        ev.line_count,
            "events_first_wall":  ev.first_wall,
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        try: pid_path.unlink()
        except Exception: pass
        try:
            if stop_flag.exists(): stop_flag.unlink()
        except Exception: pass
        print(f"[rec] done. events={ev.line_count}  "
              f"duration={meta['duration_s']:.1f}s  "
              f"video={meta['video_size_bytes']/1e6:.1f} MB")
        print(f"[rec] meta -> {meta_path}")


if __name__ == "__main__":
    main()
