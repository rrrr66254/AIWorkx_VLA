"""
Subway Surfers 전용 액션 매퍼 (NitroGen 출력 → ADB swipe).

Subway Surfers 조작:
  Swipe LEFT  → 왼쪽 레인으로 이동
  Swipe RIGHT → 오른쪽 레인으로 이동
  Swipe UP    → 점프
  Swipe DOWN  → 슬라이딩/구르기

NitroGen j_left 매핑 (18-timestep context, index 0 사용):
  X < -JOY_THRESH  →  swipe LEFT
  X > +JOY_THRESH  →  swipe RIGHT
  Y < -JOY_THRESH  →  swipe UP   (조이스틱 위쪽)
  Y > +JOY_THRESH  →  swipe DOWN

버튼 매핑 (빈도순, 활성화 시 실행):
  button[18] → swipe UP   (점프, 15% 빈도)
  button[ 5] → swipe RIGHT (11%)
  button[16] → swipe LEFT  (7%)
  button[20] → swipe LEFT  (4%)
  button[ 9] → swipe DOWN  (3%)
  button[ 2] → swipe UP    (face button / jump)
  button[ 4] → swipe LEFT  (LB)
  button[14] → swipe RIGHT
  button[ 8] → swipe UP
  button[ 7] → swipe RIGHT
  button[15] → swipe DOWN
"""
from dataclasses import dataclass

# ── 임계값 ─────────────────────────────────────────────────────
JOY_THRESH   = 0.25   # 조이스틱 활성화 임계값 (0.3 → 0.25로 낮춰 더 민감하게)
BUTTON_THRES = 0.5    # 버튼 활성화 임계값

# ── 기준 Swipe 좌표 (화면 정규화) ──────────────────────────────
# 캐릭터 위치 기준으로 swipe 시작/끝
CX = 0.50   # 화면 중앙 X
CY = 0.62   # 화면 중앙 Y (캐릭터 위치, 0.65에서 약간 위로)
DIST_H = 0.32   # 수평 swipe 거리
DIST_V = 0.25   # 수직 swipe 거리
DUR    = 120    # swipe 지속시간 ms (150→120으로 더 빠르게)

SWIPE_LEFT  = dict(type="swipe", x1=CX+DIST_H, y1=CY, x2=CX-DIST_H, y2=CY, duration_ms=DUR)
SWIPE_RIGHT = dict(type="swipe", x1=CX-DIST_H, y1=CY, x2=CX+DIST_H, y2=CY, duration_ms=DUR)
SWIPE_UP    = dict(type="swipe", x1=CX, y1=CY+DIST_V, x2=CX, y2=CY-DIST_V, duration_ms=DUR)
SWIPE_DOWN  = dict(type="swipe", x1=CX, y1=CY-DIST_V, x2=CX, y2=CY+DIST_V, duration_ms=DUR)
NOOP        = dict(type="noop")

# ── 버튼 인덱스 → 액션 (빈도 높은 순으로 정렬, 먼저 오는 것이 우선) ──
BTN_MAP = {
    18: SWIPE_UP,    # 15.0% - 가장 자주 누르는 버튼, 점프 매핑
     2: SWIPE_UP,    #  1.2% - face button (X), 점프 보조
     8: SWIPE_UP,    #  0.5% - 점프 보조
     5: SWIPE_RIGHT, # 11.2% - RB, 오른쪽 이동
     7: SWIPE_RIGHT, #  1.8%
    14: SWIPE_RIGHT, #  1.0%
    16: SWIPE_LEFT,  #  6.8% - 왼쪽 이동
    20: SWIPE_LEFT,  #  3.8%
     4: SWIPE_LEFT,  #  0.5% - LB
     9: SWIPE_DOWN,  #  3.2% - 슬라이딩
    15: SWIPE_DOWN,  #  0.8%
}


@dataclass
class ADBAction:
    type: str
    x:   float = 0.5
    y:   float = 0.5
    x2:  float = 0.5
    y2:  float = 0.5
    duration_ms: int = 120

    def to_dict(self) -> dict:
        if self.type == "tap":
            return {"type": "tap", "x": self.x, "y": self.y}
        if self.type == "long_press":
            return {"type": "long_press", "x": self.x, "y": self.y,
                    "duration_ms": self.duration_ms}
        if self.type == "swipe":
            return {"type": "swipe",
                    "x1": self.x, "y1": self.y,
                    "x2": self.x2, "y2": self.y2,
                    "duration_ms": self.duration_ms}
        return {"type": "noop"}


class ActionMapper:
    def __init__(self):
        pass

    def map(self, output) -> ADBAction:
        """NitroGen 출력 → Subway Surfers ADB swipe."""
        jl, buttons = self._parse(output)

        lx = float(jl[0])
        ly = float(jl[1])

        # 1순위: 조이스틱 (큰 값 우선)
        abs_x, abs_y = abs(lx), abs(ly)
        if max(abs_x, abs_y) >= JOY_THRESH:
            if abs_x >= abs_y:
                action_dict = SWIPE_LEFT if lx < 0 else SWIPE_RIGHT
            else:
                action_dict = SWIPE_UP if ly < 0 else SWIPE_DOWN
            return self._from_dict(action_dict)

        # 2순위: 버튼 (BTN_MAP 키 순서대로 체크)
        for btn_idx in BTN_MAP:
            if btn_idx < len(buttons) and float(buttons[btn_idx]) > BUTTON_THRES:
                return self._from_dict(BTN_MAP[btn_idx])

        return ADBAction("noop")

    # ── 내부 유틸 ──────────────────────────────────────────────

    @staticmethod
    def _parse(output):
        """NitroGen dict 또는 list 출력을 (jl, buttons)로 변환.
        Context window index 0 = 현재 스텝 액션 예측 사용.
        """
        import numpy as np
        if isinstance(output, dict):
            jl = output.get("j_left",  [[0.0, 0.0]])
            bt = output.get("buttons", [[]])
            # numpy 2D → 첫 번째 타임스텝 사용 (index 0)
            if hasattr(jl, "ndim") and jl.ndim == 2:
                jl = jl[0]
            elif isinstance(jl, list) and len(jl) > 0 and isinstance(jl[0], (list, tuple)):
                jl = jl[0]
            if hasattr(bt, "ndim") and bt.ndim == 2:
                bt = bt[0]
            elif isinstance(bt, list) and len(bt) > 0 and isinstance(bt[0], (list, tuple)):
                bt = bt[0]
            return jl, bt
        elif isinstance(output, list):
            return ([output[0], output[1]] if len(output) >= 2 else [0.0, 0.0]), output[4:]
        return [0.0, 0.0], []

    @staticmethod
    def _from_dict(d: dict) -> ADBAction:
        if d["type"] == "swipe":
            return ADBAction("swipe", d["x1"], d["y1"], d["x2"], d["y2"], d["duration_ms"])
        if d["type"] == "tap":
            return ADBAction("tap", d["x"], d["y"])
        return ADBAction("noop")
