"""
ADB interface — screen capture + touch injection.
Run directly on the server. adb must be on PATH.
"""
import subprocess, io, time, os, shutil
import numpy as np
import cv2

def _find_adb():
    if "ADB_PATH" in os.environ:
        return os.environ["ADB_PATH"]
    found = shutil.which("adb")
    if found:
        return found
    candidates = [
        os.path.expanduser("~/android-sdk/platform-tools/adb"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return "adb"

_ADB = _find_adb()


class ADBEnv:
    def __init__(self, device_serial: str = None):
        self.serial = device_serial
        self._w, self._h = None, None
        self._base = [_ADB] + (["-s", device_serial] if device_serial else [])

    # ── Internal Utilities ────────────────────────────────────

    def _run(self, args: list, timeout=10) -> str:
        try:
            result = subprocess.run(
                self._base + args,
                capture_output=True, timeout=timeout
            )
            return result.stdout.decode("utf-8", errors="ignore").strip()
        except subprocess.TimeoutExpired:
            return ""
        except Exception:
            return ""

    def _run_bytes(self, args: list, timeout=10) -> bytes:
        try:
            result = subprocess.run(
                self._base + args,
                capture_output=True, timeout=timeout
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return b""
        except Exception:
            return b""

    # ── Device Management ─────────────────────────────────────

    @staticmethod
    def list_devices() -> list[str]:
        out = subprocess.run(["adb", "devices"], capture_output=True, timeout=5)
        lines = out.stdout.decode().strip().split("\n")[1:]
        return [l.split("\t")[0] for l in lines if "\tdevice" in l]

    def is_emulator(self) -> bool:
        return self.serial is None or self.serial.startswith("emulator-")

    def wait_for_device(self, timeout=60):
        self._run(["wait-for-device"], timeout=timeout)

    # ── Screen Info ───────────────────────────────────────────

    def get_screen_size(self) -> tuple[int, int]:
        if self._w:
            return self._w, self._h
        out = self._run(["shell", "wm", "size"])
        # "Physical size: 1080x2340"
        size = out.split(":")[-1].strip()
        self._w, self._h = map(int, size.split("x"))
        return self._w, self._h

    # ── Screen Capture ────────────────────────────────────────

    def capture_screen(self) -> np.ndarray:
        for attempt in range(3):
            raw = self._run_bytes(["exec-out", "screencap", "-p"], timeout=15)
            if not raw:
                time.sleep(0.2)
                continue
            buf = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame  # BGR numpy array
            time.sleep(0.2)
        raise RuntimeError("screencap failed 3 times — check ADB connection")

    # ── Touch Injection ───────────────────────────────────────

    def _to_px(self, x_norm: float, y_norm: float) -> tuple[int, int]:
        w, h = self.get_screen_size()
        return int(x_norm * w), int(y_norm * h)

    def tap(self, x_norm: float, y_norm: float):
        px, py = self._to_px(x_norm, y_norm)
        self._run(["shell", "input", "tap", str(px), str(py)])

    def swipe(self, x1_n: float, y1_n: float, x2_n: float, y2_n: float,
              duration_ms: int = 200):
        w, h = self.get_screen_size()
        x1, y1 = int(x1_n * w), int(y1_n * h)
        x2, y2 = int(x2_n * w), int(y2_n * h)
        self._run(["shell", "input", "swipe",
                   str(x1), str(y1), str(x2), str(y2), str(duration_ms)])

    def long_press(self, x_norm: float, y_norm: float, duration_ms: int = 800):
        # swipe in place = long press
        self.swipe(x_norm, y_norm, x_norm, y_norm, duration_ms)

    # ── Action Dispatch ───────────────────────────────────────

    def execute(self, action: dict):
        t = action.get("type")
        if t == "tap":
            self.tap(action["x"], action["y"])
        elif t == "swipe":
            self.swipe(action["x1"], action["y1"],
                       action["x2"], action["y2"],
                       action.get("duration_ms", 200))
        elif t == "long_press":
            self.long_press(action["x"], action["y"],
                            action.get("duration_ms", 800))
        elif t == "noop":
            pass
        else:
            raise ValueError(f"Unknown action type: {t}")
