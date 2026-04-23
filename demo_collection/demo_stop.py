"""Gracefully stop the local demo recorder by touching stop.flag in the run dir.
Usage:
  python demo_stop_local.py            # stops the latest run in ./demos/
  python demo_stop_local.py <run_id>
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
DEMOS = os.path.join(HERE, "demos")

def main():
    if len(sys.argv) >= 2:
        run_id = sys.argv[1]
    else:
        runs = sorted(d for d in os.listdir(DEMOS)
                      if os.path.isdir(os.path.join(DEMOS, d)))
        if not runs:
            print("no runs under ./demos/"); sys.exit(2)
        run_id = runs[-1]
        print(f"[stop] using latest: {run_id}")

    run_dir = os.path.join(DEMOS, run_id)
    stop_flag = os.path.join(run_dir, "stop.flag")
    pid_file  = os.path.join(run_dir, "recorder.pid")
    if not os.path.isdir(run_dir):
        print(f"not a run dir: {run_dir}"); sys.exit(2)
    open(stop_flag, "w").write("stop")
    print(f"[stop] wrote {stop_flag}")
    # wait for recorder to acknowledge (meta.json finalized = recorder.pid removed)
    for _ in range(60):
        if not os.path.exists(pid_file):
            print("[stop] recorder exited cleanly.")
            meta = os.path.join(run_dir, "meta.json")
            if os.path.exists(meta):
                import json
                m = json.loads(open(meta, encoding="utf-8-sig").read())
                print(f"[stop] duration={m.get('duration_s','?'):.1f}s  "
                      f"events={m.get('event_count','?')}  "
                      f"video={m.get('video_size_bytes',0)/1e6:.1f} MB")
            return
        time.sleep(0.5)
    print("[stop] timeout waiting for recorder; check recorder.log")


if __name__ == "__main__":
    main()
