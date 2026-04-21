"""
NitroGen 추론 래퍼.

두 가지 모드:
  1. dummy  — 랜덤 20-dim 출력 (NitroGen 서버 없어도 파이프라인 전체 테스트)
  2. server — NitroGen inference_client.py 연결 (실제 추론)
"""
import random, time
import numpy as np
import cv2


# ── 더미 클라이언트 ──────────────────────────────────────────

class DummyNitrogenClient:
    """랜덤 액션 반환. 파이프라인 구조 테스트용."""

    def __init__(self, seed: int = None):
        self._rng = random.Random(seed)

    def infer(self, frame: np.ndarray) -> list:
        vec = [0.0] * 20
        # 조이스틱: -1.0 ~ 1.0 (NitroGen 실제 출력 범위)
        vec[0] = self._rng.uniform(-1.0, 1.0)  # LX
        vec[1] = self._rng.uniform(-1.0, 1.0)  # LY
        # 버튼: A(tap) 30% 확률, 나머지는 5%
        vec[8] = 1.0 if self._rng.random() < 0.30 else 0.0   # A → tap
        for i in [4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]:
            vec[i] = 1.0 if self._rng.random() < 0.05 else 0.0
        return vec

    def close(self):
        pass


# ── 실제 NitroGen 서버 클라이언트 ────────────────────────────

class NitrogenServerClient:
    """
    NitroGen serve.py 서버와 ZMQ로 통신.
    서버 실행: python NitroGen/scripts/serve.py ng.pt
    출력 포맷: {"j_left": [x,y], "j_right": [x,y], "buttons": [...]}
    """

    def __init__(self, host: str = "localhost", port: int = 5556):
        try:
            import sys, os
            # NitroGen이 /home/sltrain/NitroGen 또는 로컬에 있을 경우 경로 추가
            for p in ["/home/sltrain/NitroGen", os.path.expanduser("~/NitroGen")]:
                if os.path.isdir(p) and p not in sys.path:
                    sys.path.insert(0, p)
            from nitrogen.inference_client import ModelClient
            self._client = ModelClient(host=host, port=port)
        except ImportError:
            raise ImportError(
                "NitroGen이 설치되지 않았습니다.\n"
                "  git clone https://github.com/MineDojo/NitroGen\n"
                "  pip install -e 'NitroGen/[serve]'"
            )

    def infer(self, frame: np.ndarray) -> dict:
        # NitroGen 입력: 256×256 RGB
        resized = cv2.resize(frame, (256, 256))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return self._client.predict(rgb)   # dict: j_left, j_right, buttons

    def close(self):
        if hasattr(self._client, "close"):
            self._client.close()


# ── 팩토리 ───────────────────────────────────────────────────

def build_client(dummy: bool = False, host: str = "localhost", port: int = 5556):
    if dummy:
        print("[NitrogenClient] 더미 모드로 실행합니다.")
        return DummyNitrogenClient()
    print(f"[NitrogenClient] 서버 연결 시도: {host}:{port}")
    return NitrogenServerClient(host=host, port=port)
