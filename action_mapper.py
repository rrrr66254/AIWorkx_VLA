"""
Action mapper dedicated to Subway Surfers (NitroGen output -> ADB swipe).

Subway Surfers controls:
  Swipe LEFT  -> move to left lane
  Swipe RIGHT -> move to right lane
  Swipe UP    -> jump
  Swipe DOWN  -> slide/roll

NitroGen j_left mapping (18-timestep context, index 0 used):
  X < -JOY_THRESH  ->  swipe LEFT
  X > +JOY_THRESH  ->  swipe RIGHT
  Y < -JOY_THRESH  ->  swipe UP   (joystick up)
  Y > +JOY_THRESH  ->  swipe DOWN

Button mapping (sorted by frequency, active when triggered):
  button[18] -> swipe UP   (jump, 15% frequency)
  button[ 5] -> swipe RIGHT (11%)
  button[16] -> swipe LEFT  (7%)
  button[20] -> swipe LEFT  (4%)
  button[ 9] -> swipe DOWN  (3%)
  button[ 2] -> swipe UP    (face button / jump)
  button[ 4] -> swipe LEFT  (LB)
  button[14] -> swipe RIGHT
  button[ 8] -> swipe UP
  button[ 7] -> swipe RIGHT
  button[15] -> swipe DOWN
"""
from dataclasses import dataclass

# ── Thresholds ─────────────────────────────────────────────────────
JOY_THRESH   = 0.25   # joystick activation threshold (lowered from 0.3 to 0.25 for more sensitivity)
BUTTON_THRES = 0.5    # button activation threshold

# ── Reference Swipe Coordinates (normalized screen) ──────────────────────────
# Swipe start/end based on character position
CX = 0.50   # screen center X
CY = 0.62   # screen center Y (character position, slightly above 0.65)
DIST_H = 0.32   # horizontal swipe distance
DIST_V = 0.25   # vertical swipe distance
DUR    = 120    # swipe duration ms (faster, reduced from 150 to 120)

SWIPE_LEFT  = dict(type="swipe", x1=CX+DIST_H, y1=CY, x2=CX-DIST_H, y2=CY, duration_ms=DUR)
SWIPE_RIGHT = dict(type="swipe", x1=CX-DIST_H, y1=CY, x2=CX+DIST_H, y2=CY, duration_ms=DUR)
SWIPE_UP    = dict(type="swipe", x1=CX, y1=CY+DIST_V, x2=CX, y2=CY-DIST_V, duration_ms=DUR)
SWIPE_DOWN  = dict(type="swipe", x1=CX, y1=CY-DIST_V, x2=CX, y2=CY+DIST_V, duration_ms=DUR)
NOOP        = dict(type="noop")

# ── Button index -> Action (sorted by frequency descending, first match wins) ──
BTN_MAP = {
    18: SWIPE_UP,    # 15.0% - most frequently pressed button, mapped to jump
     2: SWIPE_UP,    #  1.2% - face button (X), auxiliary jump
     8: SWIPE_UP,    #  0.5% - auxiliary jump
     5: SWIPE_RIGHT, # 11.2% - RB, move right
     7: SWIPE_RIGHT, #  1.8%
    14: SWIPE_RIGHT, #  1.0%
    16: SWIPE_LEFT,  #  6.8% - move left
    20: SWIPE_LEFT,  #  3.8%
     4: SWIPE_LEFT,  #  0.5% - LB
     9: SWIPE_DOWN,  #  3.2% - slide
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
        """NitroGen output -> Subway Surfers ADB swipe."""
        jl, buttons = self._parse(output)

        lx = float(jl[0])
        ly = float(jl[1])

        # Priority 1: joystick (larger magnitude wins)
        abs_x, abs_y = abs(lx), abs(ly)
        if max(abs_x, abs_y) >= JOY_THRESH:
            if abs_x >= abs_y:
                action_dict = SWIPE_LEFT if lx < 0 else SWIPE_RIGHT
            else:
                action_dict = SWIPE_UP if ly < 0 else SWIPE_DOWN
            return self._from_dict(action_dict)

        # Priority 2: buttons (checked in BTN_MAP key order)
        for btn_idx in BTN_MAP:
            if btn_idx < len(buttons) and float(buttons[btn_idx]) > BUTTON_THRES:
                return self._from_dict(BTN_MAP[btn_idx])

        return ADBAction("noop")

    # ── Internal Utilities ──────────────────────────────────────────────

    @staticmethod
    def _parse(output):
        """Convert NitroGen dict or list output to (jl, buttons).
        Context window index 0 = use current step action prediction.
        """
        import numpy as np
        if isinstance(output, dict):
            jl = output.get("j_left",  [[0.0, 0.0]])
            bt = output.get("buttons", [[]])
            # numpy 2D -> use first timestep (index 0)
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
