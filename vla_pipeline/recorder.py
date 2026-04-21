"""
Session recorder — saves frame + action + telemetry as JSONL + PNG.
Format suitable for use as fine-tuning data later.
"""
import json, os, time
from pathlib import Path
import numpy as np
import cv2


def _numpy_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class SessionRecorder:
    def __init__(self, output_dir: str, session_id: str = None):
        sid = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(output_dir) / sid
        self.frames_dir  = self.session_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self._jsonl_path = self.session_dir / "data.jsonl"
        self._file = open(self._jsonl_path, "w", encoding="utf-8")
        self._step = 0
        print(f"[Recorder] Session started: {self.session_dir}")

    def record(self,
               frame: np.ndarray,
               action: dict,
               nitrogen_raw: list,
               telemetry: dict):
        fname = f"{self._step:06d}.png"
        cv2.imwrite(str(self.frames_dir / fname), frame)

        record = {
            "step":         self._step,
            "timestamp":    time.time(),
            "frame":        f"frames/{fname}",
            "action":       action,
            "nitrogen_raw": nitrogen_raw,
            "telemetry":    telemetry,
        }
        self._file.write(json.dumps(record, ensure_ascii=False, default=_numpy_default) + "\n")
        self._file.flush()
        self._step += 1

    def close(self):
        self._file.flush()
        self._file.close()
        print(f"[Recorder] Save complete: {self._step} steps -> {self.session_dir}")
