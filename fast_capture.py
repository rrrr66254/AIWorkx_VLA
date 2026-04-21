"""
scrcpy-based high-speed screen capture.
ADB screencap (~1fps) -> scrcpy streaming (~15-30fps)

Usage:
  cap = ScrcpyCapture(device="emulator-5554")
  cap.start()
  frame = cap.get_frame()   # numpy BGR, returns latest frame immediately
  cap.stop()
"""
import threading
import time
import numpy as np


class ScrcpyCapture:
    def __init__(
        self,
        device: str = "emulator-5554",
        max_fps: int = 30,
        max_width: int = 0,         # 0 = keep original resolution (1080x2280)
        bitrate: int = 4_000_000,   # 4Mbps - sufficient quality, low CPU
    ):
        self.device    = device
        self.max_fps   = max_fps
        self.max_width = max_width
        self.bitrate   = bitrate

        self._client  = None
        self._latest  = None
        self._lock    = threading.Lock()
        self._ready   = threading.Event()
        self._fps_count = 0
        self._fps_time  = time.time()
        self.current_fps = 0.0

    def start(self, timeout: float = 15.0):
        """Start the scrcpy client. Wait for first frame within timeout seconds."""
        import scrcpy

        self._client = scrcpy.Client(
            device=self.device,
            max_fps=self.max_fps,
            max_width=self.max_width,
            bitrate=self.bitrate,
            block_frame=False,
        )

        def on_frame(frame):
            if frame is not None:
                with self._lock:
                    self._latest = frame
                self._ready.set()
                # calculate fps
                self._fps_count += 1
                now = time.time()
                if now - self._fps_time >= 2.0:
                    self.current_fps = self._fps_count / (now - self._fps_time)
                    self._fps_count = 0
                    self._fps_time  = now

        # use add_listener API (compatible with versions without on() decorator)
        self._client.add_listener(scrcpy.EVENT_FRAME, on_frame)
        self._client.start(threaded=True)

        # wait for first frame
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError(f"[ScrcpyCapture] Failed to receive first frame within {timeout}s")

        h, w = self._latest.shape[:2]
        print(f"[ScrcpyCapture] Started: {w}x{h}  max_fps={self.max_fps}")

    def get_frame(self) -> np.ndarray:
        """Return latest frame (numpy BGR). Returns None before start() is called."""
        with self._lock:
            return self._latest

    def stop(self):
        if self._client:
            self._client.stop()
        print("[ScrcpyCapture] Stopped")
