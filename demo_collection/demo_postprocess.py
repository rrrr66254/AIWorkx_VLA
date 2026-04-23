"""Demo postprocessor.

Supports two recorder outputs:
  (A) video.mp4 (scrcpy) + events.jsonl       [new path, preferred]
  (B) frames/*.png + frame_ts.jsonl + events.jsonl   [legacy path]

Pipeline (A):
  1. Parse events.jsonl -> swipes list
  2. Extract frames from video.mp4 using OpenCV; compute per-frame wall-clock
     ts = scrcpy_wall_start + CAP_PROP_POS_MSEC/1000
  3. Optionally downsample frames (--frame-stride) and save to frames/{i:06d}.jpg
  4. Label each frame with swipe starting in (ts, ts+window]
  5. Write actions.jsonl, swipes.jsonl, summary.json, frame_ts.jsonl

Action classes: NOOP / LEFT / RIGHT / UP / DOWN / TAP
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from typing import Optional

LABEL_WINDOW_S    = 0.30   # swipe must start within this window AFTER frame ts
SWIPE_MIN_DIST_N  = 0.05   # normalized (0~1) displacement threshold for swipe vs tap
TAP_MAX_DUR_S     = 0.25

# getevent -lt line example (one of several formats):
# [  12345.678901] /dev/input/event12: EV_ABS       ABS_MT_POSITION_X    00003a5c
# [  12345.678902] /dev/input/event12: EV_ABS       ABS_MT_TRACKING_ID   000000f3
# [  12345.678903] /dev/input/event12: EV_KEY       BTN_TOUCH            DOWN
# [  12345.678904] /dev/input/event12: EV_SYN       SYN_REPORT           00000000


def parse_line(raw: str):
    """Return (dev, ev_type, code, value) or None."""
    # Strip the "[  ts ]" prefix if present
    s = raw.strip()
    if s.startswith("["):
        rb = s.find("]")
        if rb < 0: return None
        s = s[rb+1:].strip()
    # Now: "/dev/input/eventN: EV_ABS ABS_MT_POSITION_X 00003a5c"
    if ":" not in s: return None
    dev, rest = s.split(":", 1)
    parts = rest.split()
    if len(parts) < 3: return None
    ev_type, code, value = parts[0], parts[1], parts[2]
    return dev.strip(), ev_type, code, value


def hex_or_int(v: str) -> int:
    try:
        return int(v, 16)
    except Exception:
        try: return int(v)
        except Exception: return 0


def extract_swipes(events_path: Path, screen_w: int, screen_h: int,
                   abs_max_x: int = 32767, abs_max_y: int = 32767):
    """Scan events.jsonl and emit swipe/tap records.

    Returns list of dicts:
      {t_start, t_end, dur_s, dev, slot, tracking_id,
       x1n, y1n, x2n, y2n, dxn, dyn, dist, cls}
    """
    # Per-device, per-slot tracker
    # state[dev][slot] = {tracking_id, last_x, last_y, start_x, start_y, t_start, t_last, active}
    state: dict = {}
    current_slot: dict = {}  # dev -> current slot index (default 0)
    swipes = []

    def start_track(dev, slot, x, y, t):
        state.setdefault(dev, {})
        state[dev][slot] = {
            "t_start": t, "t_last": t,
            "sx": x, "sy": y, "lx": x, "ly": y,
            "active": True,
        }

    def close_track(dev, slot):
        st = state.get(dev, {}).get(slot)
        if not st or not st["active"]:
            return
        st["active"] = False
        dx = st["lx"] - st["sx"]
        dy = st["ly"] - st["sy"]
        x1n = st["sx"] / abs_max_x
        y1n = st["sy"] / abs_max_y
        x2n = st["lx"] / abs_max_x
        y2n = st["ly"] / abs_max_y
        dxn = x2n - x1n
        dyn = y2n - y1n
        dist = (dxn*dxn + dyn*dyn) ** 0.5
        dur  = st["t_last"] - st["t_start"]
        # classify
        if dist < SWIPE_MIN_DIST_N and dur < TAP_MAX_DUR_S:
            cls = "TAP"
        elif abs(dxn) >= abs(dyn):
            cls = "LEFT" if dxn < 0 else "RIGHT"
        else:
            cls = "UP" if dyn < 0 else "DOWN"
        swipes.append({
            "t_start": st["t_start"], "t_end": st["t_last"], "dur_s": dur,
            "dev": dev, "slot": slot,
            "x1n": x1n, "y1n": y1n, "x2n": x2n, "y2n": y2n,
            "dxn": dxn, "dyn": dyn, "dist": dist, "cls": cls,
        })

    with open(events_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            wall = float(obj.get("wall", 0.0))
            parsed = parse_line(obj.get("raw", ""))
            if not parsed: continue
            dev, ev_type, code, value = parsed

            if code == "ABS_MT_SLOT":
                current_slot[dev] = hex_or_int(value)

            elif code == "ABS_MT_TRACKING_ID":
                v = hex_or_int(value)
                slot = current_slot.get(dev, 0)
                if value.lower() == "ffffffff" or v == 0xFFFFFFFF:
                    # finger lift for this slot
                    close_track(dev, slot)
                else:
                    # new contact on this slot
                    state.setdefault(dev, {})
                    # we don't have coordinates yet; they'll arrive; mark seeded
                    state[dev][slot] = {
                        "t_start": wall, "t_last": wall,
                        "sx": None, "sy": None, "lx": None, "ly": None,
                        "active": True,
                    }

            elif code in ("ABS_MT_POSITION_X", "ABS_MT_POSITION_Y"):
                slot = current_slot.get(dev, 0)
                st = state.get(dev, {}).get(slot)
                if not st or not st["active"]:
                    # start implicit track
                    start_track(dev, slot, 0, 0, wall)
                    st = state[dev][slot]
                v = hex_or_int(value)
                if code == "ABS_MT_POSITION_X":
                    if st["sx"] is None: st["sx"] = v
                    st["lx"] = v
                else:
                    if st["sy"] is None: st["sy"] = v
                    st["ly"] = v
                st["t_last"] = wall

            elif code == "BTN_TOUCH" and value == "UP":
                # finger up on whatever the current slot is
                slot = current_slot.get(dev, 0)
                close_track(dev, slot)

    # flush remaining (shouldn't happen often)
    for dev, slots in list(state.items()):
        for slot in list(slots.keys()):
            if slots[slot].get("active") and slots[slot].get("sx") is not None:
                close_track(dev, slot)

    # filter out tracks that never got coordinates
    swipes = [s for s in swipes if s["x1n"] is not None and s["x2n"] is not None]
    swipes.sort(key=lambda s: s["t_start"])
    return swipes


def label_frames(frame_ts_path: Path, swipes: list, window: float):
    """For each frame, attach the action class of the first swipe starting in
    (frame_ts, frame_ts + window]. Unclaimed frames -> NOOP.

    A swipe is attached to the EARLIEST eligible frame only; subsequent frames
    during the same swipe get NOOP to avoid duplicate-labeling the motion.
    """
    frames = []
    with open(frame_ts_path, "r", encoding="utf-8") as f:
        for ln in f:
            try: frames.append(json.loads(ln))
            except Exception: pass
    frames.sort(key=lambda x: x["ts"])

    # Sort swipes already done. Assign each swipe to earliest frame with ts <= t_start
    sw_idx = 0
    used = [False] * len(swipes)
    labels = []
    for fr in frames:
        fts = fr["ts"]
        assigned = None
        # advance sw_idx to first swipe with t_start > fts - tiny
        while sw_idx < len(swipes) and swipes[sw_idx]["t_start"] < fts:
            # already in the past; skip (would have been assigned to earlier frame)
            if not used[sw_idx]:
                # swipe started slightly before this frame — consume it anyway
                used[sw_idx] = True
                assigned = swipes[sw_idx]
                sw_idx += 1
                break
            sw_idx += 1
        if assigned is None:
            # look ahead within window
            j = sw_idx
            while j < len(swipes) and swipes[j]["t_start"] <= fts + window:
                if not used[j]:
                    used[j] = True
                    assigned = swipes[j]
                    break
                j += 1
        cls = assigned["cls"] if assigned else "NOOP"
        rec = {
            "i": fr["i"],
            "ts": fts,
            "path": fr["path"],
            "action": cls,
        }
        if assigned:
            rec["swipe"] = {k: assigned[k] for k in
                            ("x1n","y1n","x2n","y2n","dur_s","dist","t_start","t_end")}
        labels.append(rec)
    return labels


def extract_frames_from_video(mp4_path: Path, frames_dir: Path,
                              scrcpy_wall_start: float,
                              stride: int = 1, max_frames: int = 0,
                              jpg_quality: int = 90,
                              frame_size: int = 0) -> list[dict]:
    """Extracts every `stride`-th frame from the mp4. Returns list of frame metadata:
       [{"i": <extracted_idx>, "src_idx": <mp4_frame_idx>, "ts": <wall_time>,
         "pos_msec": <relative>, "path": "frames/000001.jpg"}]"""
    import cv2
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps   = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    print(f"[post] video: total_frames~={total}  fps~={fps:.2f}")
    out = []; src_idx = 0; kept = 0
    while True:
        ok = cap.grab()
        if not ok: break
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if src_idx % stride == 0:
            ret, frame = cap.retrieve()
            if not ret: break
            if frame_size > 0:
                h0, w0 = frame.shape[:2]
                m = max(h0, w0)
                if m > frame_size:
                    scale = frame_size / m
                    frame = cv2.resize(frame, (int(w0*scale), int(h0*scale)),
                                       interpolation=cv2.INTER_AREA)
            kept += 1
            name = f"{kept:06d}.jpg"
            cv2.imwrite(str(frames_dir / name), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
            out.append({
                "i":        kept,
                "src_idx":  src_idx,
                "pos_msec": pos_msec,
                "ts":       scrcpy_wall_start + pos_msec / 1000.0,
                "path":     f"frames/{name}",
            })
            if kept % 500 == 0:
                print(f"[post] extracted {kept} frames ...")
            if max_frames and kept >= max_frames: break
        src_idx += 1
    cap.release()
    print(f"[post] extracted {kept} frames (from {src_idx} decoded)")
    return out


def label_frames_inline(frame_records: list[dict], swipes: list, window: float):
    """Variant of label_frames that takes a preloaded list (not a file)."""
    frame_records = sorted(frame_records, key=lambda x: x["ts"])
    sw_idx = 0
    used = [False] * len(swipes)
    labels = []
    for fr in frame_records:
        fts = fr["ts"]
        assigned = None
        while sw_idx < len(swipes) and swipes[sw_idx]["t_start"] < fts:
            if not used[sw_idx]:
                used[sw_idx] = True
                assigned = swipes[sw_idx]
                sw_idx += 1
                break
            sw_idx += 1
        if assigned is None:
            j = sw_idx
            while j < len(swipes) and swipes[j]["t_start"] <= fts + window:
                if not used[j]:
                    used[j] = True; assigned = swipes[j]; break
                j += 1
        cls = assigned["cls"] if assigned else "NOOP"
        rec = {"i": fr["i"], "ts": fts, "path": fr["path"], "action": cls}
        if assigned:
            rec["swipe"] = {k: assigned[k] for k in
                            ("x1n","y1n","x2n","y2n","dur_s","dist","t_start","t_end")}
        labels.append(rec)
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="dir containing video.mp4 + events.jsonl + meta.json (can be NAS path)")
    ap.add_argument("--out-dir", default=None,
                    help="dir to write frames/, actions.jsonl, etc. (default = run_dir)")
    ap.add_argument("--window",        type=float, default=LABEL_WINDOW_S)
    ap.add_argument("--frame-stride",  type=int,   default=1,
                    help="keep every N-th video frame (1 = all, 3 = 10fps if video is 30fps)")
    ap.add_argument("--frame-size",    type=int,   default=0,
                    help="downscale longest edge of each frame to N px (0 = native)")
    ap.add_argument("--max-frames",    type=int,   default=0,
                    help="extract at most N frames (0 = no cap)")
    ap.add_argument("--jpg-quality",   type=int,   default=90)
    ap.add_argument("--skip-frame-extract", action="store_true",
                    help="skip video->frames step; use existing frame_ts.jsonl")
    ap.add_argument("--start-s", type=float, default=0.0,
                    help="drop frames/swipes BEFORE this many seconds from scrcpy_wall_start")
    ap.add_argument("--end-s",   type=float, default=0.0,
                    help="drop frames/swipes AFTER this many seconds (0 = no cap)")
    args = ap.parse_args()
    root = Path(args.run_dir)
    out  = Path(args.out_dir) if args.out_dir else root
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((root / "meta.json").read_text(encoding="utf-8-sig"))
    screen_w, screen_h = meta["screen_w"], meta["screen_h"]
    abs_max_x = meta.get("abs_max_x", meta.get("abs_max", 32767))
    abs_max_y = meta.get("abs_max_y", meta.get("abs_max", 32767))
    print(f"[post] screen={screen_w}x{screen_h}  abs=({abs_max_x},{abs_max_y})  "
          f"touch='{meta.get('touch_name','?')}'")

    # swipes first
    swipes = extract_swipes(root / "events.jsonl", screen_w, screen_h,
                            abs_max_x, abs_max_y)
    print(f"[post] swipes extracted: {len(swipes)}")

    # trim by wall-clock window if requested
    scrcpy_start_wall = meta.get("scrcpy_wall_start") or \
                        meta.get("events_first_wall") or 0.0
    t_lo = scrcpy_start_wall + args.start_s
    t_hi = scrcpy_start_wall + args.end_s if args.end_s > 0 else float("inf")
    if args.start_s > 0 or args.end_s > 0:
        before = len(swipes)
        swipes = [s for s in swipes if t_lo <= s["t_start"] <= t_hi]
        print(f"[post] trimmed swipes: {before} -> {len(swipes)} "
              f"(window [{args.start_s:.1f}s, {args.end_s:.1f}s])")
    cls_count = {}
    for s in swipes:
        cls_count[s["cls"]] = cls_count.get(s["cls"], 0) + 1
    print(f"[post] swipe classes: {cls_count}")

    # frames: from mp4 (new) or frame_ts.jsonl (legacy)
    video_path = root / meta.get("video_path", "video.mp4")
    if video_path.exists() and not args.skip_frame_extract:
        scrcpy_start = meta.get("scrcpy_wall_start") or \
                       meta.get("events_first_wall") or 0.0
        frame_records = extract_frames_from_video(
            video_path, out / "frames", scrcpy_start,
            stride=args.frame_stride, max_frames=args.max_frames,
            jpg_quality=args.jpg_quality, frame_size=args.frame_size,
        )
        # persist frame_ts
        with open(out / "frame_ts.jsonl", "w", encoding="utf-8") as f:
            for fr in frame_records:
                f.write(json.dumps(fr, ensure_ascii=False) + "\n")
        # also trim frames by same window
        if args.start_s > 0 or args.end_s > 0:
            frame_records = [fr for fr in frame_records if t_lo <= fr["ts"] <= t_hi]
            print(f"[post] trimmed frames to {len(frame_records)}")
        labels = label_frames_inline(frame_records, swipes, args.window)
    else:
        labels = label_frames(out / "frame_ts.jsonl", swipes, args.window)

    label_count = {}
    for l in labels:
        label_count[l["action"]] = label_count.get(l["action"], 0) + 1
    print(f"[post] frame labels: total={len(labels)}  {label_count}")

    out_actions = out / "actions.jsonl"
    with open(out_actions, "w", encoding="utf-8") as f:
        for l in labels:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    out_swipes = out / "swipes.jsonl"
    with open(out_swipes, "w", encoding="utf-8") as f:
        for s in swipes:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    # copy meta.json into out (small; useful to keep with frames)
    (out / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "run_dir":       str(root),
        "out_dir":       str(out),
        "duration_s":    meta.get("duration_s"),
        "frame_count":   len(labels),
        "swipe_count":   len(swipes),
        "swipe_class":   cls_count,
        "label_class":   label_count,
        "frame_stride":  args.frame_stride,
        "frame_size":    args.frame_size,
        "video_used":    video_path.exists() and not args.skip_frame_extract,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[post] wrote {out_actions}")
    print(f"[post] wrote {out_swipes}")
    print(f"[post] summary -> {out/'summary.json'}")


if __name__ == "__main__":
    main()
