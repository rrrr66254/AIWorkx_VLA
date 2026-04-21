"""
NitroGen inference wrapper.

Two modes:
  1. dummy  -- random 20-dim output (tests full pipeline without a NitroGen server)
  2. server -- connects to NitroGen inference_client.py (real inference)
"""
import random, time
import numpy as np
import cv2


# ── Dummy Client ─────────────────────────────────────────────

class DummyNitrogenClient:
    """Returns random actions. For testing pipeline structure."""

    def __init__(self, seed: int = None):
        self._rng = random.Random(seed)

    def infer(self, frame: np.ndarray) -> list:
        vec = [0.0] * 20
        # joystick: -1.0 ~ 1.0 (actual NitroGen output range)
        vec[0] = self._rng.uniform(-1.0, 1.0)  # LX
        vec[1] = self._rng.uniform(-1.0, 1.0)  # LY
        # buttons: A(tap) 30% probability, others 5%
        vec[8] = 1.0 if self._rng.random() < 0.30 else 0.0   # A -> tap
        for i in [4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]:
            vec[i] = 1.0 if self._rng.random() < 0.05 else 0.0
        return vec

    def close(self):
        pass


# ── Real NitroGen Server Client ───────────────────────────────

class NitrogenServerClient:
    """
    Communicates with NitroGen serve.py server via ZMQ.
    Start server: python NitroGen/scripts/serve.py ng.pt
    Output format: {"j_left": [x,y], "j_right": [x,y], "buttons": [...]}
    """

    def __init__(self, host: str = "localhost", port: int = 5556):
        try:
            import sys, os
            nitrogen_path = os.path.expanduser("~/NitroGen")
            if os.path.isdir(nitrogen_path) and nitrogen_path not in sys.path:
                sys.path.insert(0, nitrogen_path)
            from nitrogen.inference_client import ModelClient
            self._client = ModelClient(host=host, port=port)
        except ImportError:
            raise ImportError(
                "NitroGen is not installed.\n"
                "  git clone https://github.com/MineDojo/NitroGen\n"
                "  pip install -e 'NitroGen/[serve]'"
            )

    def infer(self, frame: np.ndarray) -> dict:
        # NitroGen input: 256x256 RGB
        resized = cv2.resize(frame, (256, 256))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return self._client.predict(rgb)   # dict: j_left, j_right, buttons

    def close(self):
        if hasattr(self._client, "close"):
            self._client.close()


# ── Factory ───────────────────────────────────────────────────

def build_client(dummy: bool = False, host: str = "localhost", port: int = 5556):
    if dummy:
        print("[NitrogenClient] Running in dummy mode.")
        return DummyNitrogenClient()
    print(f"[NitrogenClient] Attempting server connection: {host}:{port}")
    return NitrogenServerClient(host=host, port=port)
