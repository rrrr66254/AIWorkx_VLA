"""
scrcpy 기반 고속 화면 캡처.
ADB screencap(~1fps) → scrcpy 스트리밍(~15-30fps)

사용:
  cap = ScrcpyCapture(device="emulator-5554")
  cap.start()
  frame = cap.get_frame()   # numpy BGR, 최신 프레임 즉시 반환
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
        max_width: int = 0,         # 0 = 원본 해상도 유지 (1080×2280)
        bitrate: int = 4_000_000,   # 4Mbps - 충분한 화질, 적은 CPU
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
        """scrcpy 클라이언트 시작. timeout 초 내에 첫 프레임 대기."""
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
                # fps 계산
                self._fps_count += 1
                now = time.time()
                if now - self._fps_time >= 2.0:
                    self.current_fps = self._fps_count / (now - self._fps_time)
                    self._fps_count = 0
                    self._fps_time  = now

        # add_listener API 사용 (on() 데코레이터 없는 버전 대응)
        self._client.add_listener(scrcpy.EVENT_FRAME, on_frame)
        self._client.start(threaded=True)

        # 첫 프레임 대기
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError(f"[ScrcpyCapture] {timeout}초 내 첫 프레임 수신 실패")

        h, w = self._latest.shape[:2]
        print(f"[ScrcpyCapture] 시작: {w}×{h}  max_fps={self.max_fps}")

    def get_frame(self) -> np.ndarray:
        """최신 프레임 반환 (numpy BGR). start() 이전엔 None."""
        with self._lock:
            return self._latest

    def stop(self):
        if self._client:
            self._client.stop()
        print("[ScrcpyCapture] 종료")
