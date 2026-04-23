# Subway Surfers Demo Collection

Local tooling to record Android gameplay (video + raw touch events) and convert
it into labeled frame-action pairs for training vision-language-action (VLA) models.

Built around **scrcpy** (for high-fps video) + **adb getevent** (for millisecond
touch traces) + **OpenCV** (for offline frame extraction). Captures ~30 fps
H.264 video and nanosecond-accurate swipe coordinates, then aligns them.

## What you get per session

```
demos/20260423_143927/
  video.mp4          # scrcpy H.264 recording (~1 GB for 15 min)
  events.jsonl       # raw adb getevent lines with wall-clock timestamps
  meta.json          # device info, touch calibration, start/end times
  recorder.log       # scrcpy stdout/stderr

After postprocess:
  frames/000001.jpg  # downsampled frames (default: longest edge 384px, 10fps)
  frame_ts.jsonl     # per-frame wall-clock timestamp
  swipes.jsonl       # parsed touch tracks, each classified as TAP/LEFT/RIGHT/UP/DOWN
  actions.jsonl      # frame -> action labels (for BC/VLA training)
  summary.json       # quick stats
```

`actions.jsonl` is the training file — each line is one frame with its action label.

## Prerequisites

- **Python 3.9+**
- **Android phone** with USB debugging enabled
- **adb** (Android platform-tools) — https://developer.android.com/tools/releases/platform-tools
- **scrcpy** ≥ 2.0 — https://github.com/Genymobile/scrcpy

Put `adb` and `scrcpy` on PATH, or set `ADB_PATH` / `SCRCPY_PATH` env vars.

Check with:
```bash
adb devices
scrcpy --version
```

## Install

```bash
git clone https://github.com/rrrr66254/AIWorkx_VLA.git
cd AIWorkx_VLA/demo_collection
pip install -r requirements.txt
```

## One-shot usage

```bash
# 1) Start recording (plug phone, unlock, accept RSA prompt once)
python demo_record.py

# 2) Play your game on the phone

# 3) Stop cleanly (writes meta.json, finalizes mp4)
python demo_stop.py          # stops the latest run

# 4) Convert video -> labeled frames + actions
python demo_postprocess.py demos/<run_id> \
       --frame-stride 3 --frame-size 384 --jpg-quality 88

# 5) Inspect data quality
python demo_inspect.py                 # all runs
python demo_inspect.py <run_id>        # one run
```

### Recommended: enable airplane mode on the phone

Ads during gameplay will pollute the data (the "close ad" taps get labeled as
game actions). Subway Surfers plays offline after the first launch — airplane
mode fully prevents ads.

### Trimming ad segments

If you realize an ad appeared near the end of a session, re-run postprocess
with `--end-s <seconds>` to cut it:

```bash
# keep only the first 790 seconds of this run
python demo_postprocess.py demos/<run_id> --end-s 790 \
       --frame-stride 3 --frame-size 384
```

`demo_inspect.py` helps find the cutoff — the script prints the last 15 swipes
of each run; ads usually show as rapid TAPs in screen corners (typical ad-close
X button coordinates).

## Script reference

| Script | Purpose |
|--------|---------|
| `demo_record.py`       | Launcher: finds adb/scrcpy, picks device, spawns recorder |
| `demo_recorder.py`     | Worker: runs scrcpy + `adb getevent`, writes to `demos/<run_id>/` |
| `demo_stop.py`         | Creates `stop.flag` in latest run; waits for graceful exit |
| `demo_postprocess.py`  | mp4 → frames + events → swipes + action labels |
| `demo_inspect.py`      | Prints per-run stats + swipe class distribution |
| `demo_upload.py`       | *(optional)* SFTP raw to NAS + run postprocess on a remote box |

### Record options

```
python demo_record.py [max_fps]
  max_fps                 override default 30
  --device <serial>       pick specific device (else first device)
  --out <dir>             output root (default ./demos)
```

Environment:
- `ADB_PATH`    full path to `adb` binary
- `SCRCPY_PATH` full path to `scrcpy` binary

### Postprocess options

```
python demo_postprocess.py <run_dir> [flags]
  --out-dir <dir>         write outputs to a different dir (keep raw separate)
  --frame-stride N        keep every N-th video frame (default 1; stride=3 on 30fps -> 10fps)
  --frame-size N          downscale longest edge to N px (default 0 = native)
  --jpg-quality N         JPEG quality 1-100 (default 90)
  --start-s S             drop frames/swipes BEFORE this time (relative to scrcpy start)
  --end-s S               drop frames/swipes AFTER this time (0 = no cap)
  --skip-frame-extract    reuse existing frame_ts.jsonl (re-label only)
  --window S              swipe-to-frame assignment window (default 0.30s)
```

### Action classification

A touch track becomes a swipe if dx²+dy² > `SWIPE_MIN_DIST_N` (normalized
0..1). Otherwise it's a `TAP`. Direction is taken from the dominant axis.

| Class  | Condition |
|--------|-----------|
| TAP    | normalized distance < 0.05 AND duration < 0.25 s |
| LEFT   | dxn < 0 AND \|dxn\| ≥ \|dyn\| |
| RIGHT  | dxn > 0 AND \|dxn\| ≥ \|dyn\| |
| UP     | dyn < 0 AND \|dyn\| > \|dxn\| |
| DOWN   | dyn > 0 AND \|dyn\| > \|dxn\| |
| NOOP   | frame that has no swipe starting within 0.30 s |

Thresholds live at the top of `demo_postprocess.py`.

## Optional: remote upload

If you have a shared NAS + GPU server and want to push raw recordings to NAS
and only keep downsampled frames on the GPU box, copy and fill the template:

```bash
cp config.example.py config.py
# edit SERVER_HOST/USER/PASS and the NAS_BASE / SERVER_BASE paths
python demo_upload.py <run_id> --delete-local
```

`demo_upload.py`:
1. SFTPs `video.mp4 / events.jsonl / meta.json / recorder.log` to `NAS_BASE/<run_id>/`
2. SSHes in and runs `demo_postprocess.py` with `--frame-stride 3 --frame-size 384`, writing outputs to `SERVER_BASE/<run_id>/`
3. With `--delete-local`, removes the local 1 GB mp4 after successful upload

`config.py` is git-ignored. `config.example.py` is the template.

## File sizes (rough, 15 min session @ 1080x2340)

| Artifact | Size |
|----------|------|
| `video.mp4` (scrcpy 8 Mbps)        | ~1.1 GB |
| `events.jsonl`                     | ~8 MB   |
| `frames/` (384px, stride=3, q=88)  | ~215 MB |
| `actions.jsonl` + `swipes.jsonl`   | ~1 MB   |

So you can keep raw mp4s on a NAS and only 215 MB per session on your training disk.

## Troubleshooting

**`adb devices` shows nothing**
- Unplug + replug USB cable
- Re-enable USB debugging in *Developer options*
- If still nothing, revoke USB debugging authorizations on the phone and accept the RSA prompt again
- USB mode must be **File transfer (MTP)**, not charging-only

**scrcpy dies immediately**
- Check `demos/<run_id>/recorder.log` — common causes: wrong scrcpy version, phone screen off, another scrcpy running

**Frames extracted count ≠ expected**
- Normal: scrcpy uses variable frame rate. `total_frames / stride` is approximate.
- Check `meta.json:duration_s` × target fps vs. extracted count.

**Touch device detection picks wrong input**
- Phone has both `sec_touchscreen` and `sec_touchpad`. The recorder scores candidates and prefers names containing "touchscreen". Override by passing a pre-built meta.json or editing `detect_touch_abs_max()`.

## License & disclaimer

For research use. Do not commit your `config.py`, recorded `demos/`, or any
app `.apk` to the repo — `.gitignore` handles this.

The recorder captures raw screen content and touch coordinates; record only
gameplay of apps you are authorized to record.
