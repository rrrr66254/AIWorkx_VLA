"""Inspect local demo data quality.

Usage:
  python demo_inspect.py                 # inspect all runs under ./demos
  python demo_inspect.py <run_id>        # inspect a single run
  python demo_inspect.py --root <dir>    # inspect runs under a different dir

Reports per run:
  duration, frame count / fps, swipe count & class distribution,
  label distribution (NOOP ratio), action rate (swipes/s),
  swipe duration quantiles, swipe distance quantiles,
  spatial coverage (where on screen the swipes start).
And an aggregate total across all runs.

Requires that `demo_postprocess.py` has already been run on each run
(summary.json + swipes.jsonl must exist).
"""
import sys, io, os, json, argparse

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))


def pct(n, d):
    return f"{(100.0*n/d):.1f}%" if d else "-"


def quantiles(xs, qs=(0.25, 0.5, 0.75, 0.95)):
    if not xs: return [None]*len(qs)
    xs = sorted(xs); n = len(xs)
    out = []
    for q in qs:
        k = min(n-1, max(0, int(q*n)))
        out.append(xs[k])
    return out


def grid_hist(points, bins=(3, 6)):
    bx, by = bins
    g = [[0]*bx for _ in range(by)]
    for x, y in points:
        if x is None or y is None: continue
        i = min(bx-1, max(0, int(x*bx)))
        j = min(by-1, max(0, int(y*by)))
        g[j][i] += 1
    return g


def inspect_run(root, run_id):
    rdir = os.path.join(root, run_id)
    print(f"\n=== {run_id} ===")
    sp = os.path.join(rdir, "summary.json")
    mp = os.path.join(rdir, "meta.json")
    if not os.path.exists(sp) or not os.path.exists(mp):
        print(f"  [skip] missing summary.json / meta.json (run `demo_postprocess.py` first)")
        return None
    try:
        summary = json.loads(open(sp, encoding="utf-8-sig").read())
        meta    = json.loads(open(mp, encoding="utf-8-sig").read())
    except Exception as e:
        print(f"  [skip] read error: {e}"); return None

    dur = summary.get("duration_s") or meta.get("duration_s") or 0
    n_frames = summary.get("frame_count", 0)
    n_sw     = summary.get("swipe_count", 0)
    lbl      = summary.get("label_class", {})
    cls      = summary.get("swipe_class", {})

    print(f"  duration: {dur:.1f}s  frames: {n_frames} ({n_frames/max(dur,1):.2f} fps saved)")
    print(f"  screen:   {meta.get('screen_w')}x{meta.get('screen_h')}  touch='{meta.get('touch_name','?')}'")
    print(f"  swipes:   {n_sw}  ({n_sw/max(dur,1):.2f}/s)  classes: {cls}")
    print(f"  labels:   NOOP={lbl.get('NOOP',0)} ({pct(lbl.get('NOOP',0), n_frames)} of frames)")

    durs, dists, starts = [], [], []
    swp = os.path.join(rdir, "swipes.jsonl")
    if os.path.exists(swp):
        with open(swp, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if not ln.strip(): continue
                try: s = json.loads(ln)
                except Exception: continue
                durs.append(s.get("dur_s", 0.0))
                dists.append(s.get("dist", 0.0))
                starts.append((s.get("x1n"), s.get("y1n")))
    if durs:
        qd = quantiles(durs); qdi = quantiles(dists)
        print(f"  swipe dur (s): p25={qd[0]:.2f}  p50={qd[1]:.2f}  p75={qd[2]:.2f}  p95={qd[3]:.2f}")
        print(f"  swipe dist:    p25={qdi[0]:.2f}  p50={qdi[1]:.2f}  p75={qdi[2]:.2f}  p95={qdi[3]:.2f}")
    g = grid_hist(starts, (3, 6))
    print(f"  swipe start grid (3x6, row0=top):")
    for row in g:
        tot = sum(row)
        print("    " + "  ".join(f"{v:4d}" for v in row) + f"   |{tot}")

    warns = []
    if n_sw == 0: warns.append("ZERO swipes - getevent not capturing?")
    if lbl.get("NOOP", 0) / max(n_frames,1) > 0.95: warns.append("NOOP > 95% - sparse actions")
    if cls and max(cls.values()) / max(sum(cls.values()), 1) > 0.7:
        warns.append("one class dominates (>70%) - biased")
    if n_frames / max(dur,1) < 8: warns.append(f"saved fps < 8 ({n_frames/max(dur,1):.1f}) - stride too high?")
    if warns:
        print("  [!] " + "  |  ".join(warns))
    else:
        print("  [ok] looks healthy")
    return {"run_id": run_id, "dur": dur, "frames": n_frames, "swipes": n_sw,
            "cls": cls, "noop": lbl.get("NOOP", 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?", default=None)
    ap.add_argument("--root", default=os.path.join(HERE, "demos"))
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"[err] not a directory: {root}"); sys.exit(2)

    if args.run_id:
        ids = [args.run_id]
    else:
        ids = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)))
    print(f"[inspect] {len(ids)} run(s) under {root}")

    rows = []
    for rid in ids:
        r = inspect_run(root, rid)
        if r: rows.append(r)

    if len(rows) >= 2:
        td = sum(r["dur"] for r in rows)
        tf = sum(r["frames"] for r in rows)
        ts = sum(r["swipes"] for r in rows)
        tc = {}
        for r in rows:
            for k, v in r["cls"].items(): tc[k] = tc.get(k, 0) + v
        print(f"\n=== TOTAL ({len(rows)} runs) ===")
        print(f"  duration: {td/60:.1f} min  frames: {tf}  swipes: {ts}  classes: {tc}")
        print(f"  rate:     {ts/max(td,1):.2f} swipes/s")


if __name__ == "__main__":
    main()
