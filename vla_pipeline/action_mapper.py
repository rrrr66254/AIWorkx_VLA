"""
NitroGen gamepad output (20-dim) -> ADB touch action conversion.

NitroGen output structure (20-dim):
  [0]  LX  (-32767 ~ 32767)  Left stick X
  [1]  LY  (-32767 ~ 32767)  Left stick Y
  [2]  RX  (-32767 ~ 32767)  Right stick X
  [3]  RY  (-32767 ~ 32767)  Right stick Y
  [4]  DPAD_UP     (0/1)
  [5]  DPAD_DOWN   (0/1)
  [6]  DPAD_LEFT   (0/1)
  [7]  DPAD_RIGHT  (0/1)
  [8]  A           (0/1)   -> tap
  [9]  B           (0/1)   -> long press
  [10] X           (0/1)
  [11] Y           (0/1)
  [12] LB          (0/1)
  [13] RB          (0/1)
  [14] LT  (0~1 continuous)
  [15] RT  (0~1 continuous)
  [16] L3          (0/1)
  [17] R3          (0/1)
  [18] START       (0/1)
  [19] BACK        (0/1)

Mapping strategy (cursor-based):
  LX, LY -> virtual cursor movement on screen
  A      -> tap at cursor position
  B      -> long press at cursor position
  D-pad  -> directional swipe (fixed distance 0.3)
  others -> unused (noop)
"""
import math
from dataclasses import dataclass

JOYSTICK_MAX  = 1.0
DEAD_ZONE     = 0.2    # cursor does not move below this magnitude
CURSOR_SPEED  = 0.04   # cursor movement ratio per step
SWIPE_DIST    = 0.30   # D-pad swipe distance (screen ratio)
BUTTON_THRES  = 0.5    # button activation threshold


@dataclass
class ADBAction:
    type: str          # "tap" | "swipe" | "long_press" | "noop"
    x:    float = 0.5  # normalized coordinates [0,1]
    y:    float = 0.5
    x2:   float = 0.5  # swipe endpoint
    y2:   float = 0.5
    duration_ms: int = 200

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
        self.cursor_x = 0.5
        self.cursor_y = 0.5

    def map(self, output: list | dict) -> ADBAction:
        """
        NitroGen 20-dim output -> ADBAction.
        output: list[float] or dict (supports both NitroGen inference result formats)
        """
        vec = self._to_vec(output)

        lx = vec[0] / JOYSTICK_MAX
        ly = vec[1] / JOYSTICK_MAX
        btn_a   = vec[8]  > BUTTON_THRES
        btn_b   = vec[9]  > BUTTON_THRES
        dpad_u  = vec[4]  > BUTTON_THRES
        dpad_d  = vec[5]  > BUTTON_THRES
        dpad_l  = vec[6]  > BUTTON_THRES
        dpad_r  = vec[7]  > BUTTON_THRES

        # cursor movement
        mag = math.sqrt(lx**2 + ly**2)
        if mag > DEAD_ZONE:
            self.cursor_x = max(0.0, min(1.0, self.cursor_x + lx * CURSOR_SPEED))
            self.cursor_y = max(0.0, min(1.0, self.cursor_y + ly * CURSOR_SPEED))

        # D-pad -> swipe (higher priority)
        cx, cy = self.cursor_x, self.cursor_y
        if dpad_u:
            return ADBAction("swipe", cx, cy, cx, max(0.0, cy - SWIPE_DIST), 200)
        if dpad_d:
            return ADBAction("swipe", cx, cy, cx, min(1.0, cy + SWIPE_DIST), 200)
        if dpad_l:
            return ADBAction("swipe", cx, cy, max(0.0, cx - SWIPE_DIST), cy, 200)
        if dpad_r:
            return ADBAction("swipe", cx, cy, min(1.0, cx + SWIPE_DIST), cy, 200)

        # buttons
        if btn_b:
            return ADBAction("long_press", cx, cy, duration_ms=800)
        if btn_a:
            return ADBAction("tap", cx, cy)

        return ADBAction("noop")

    # ── Format Normalization ──────────────────────────────────

    @staticmethod
    def _to_vec(output) -> list:
        if isinstance(output, list):
            return output + [0.0] * max(0, 20 - len(output))

        # NitroGen inference_client.py output: {j_left:(N,2), j_right:(N,2), buttons:(N,M)}
        # use only the first timestep out of N action horizon predictions
        if isinstance(output, dict):
            import numpy as np
            vec = [0.0] * 20
            jl = output.get("j_left",  [0.0, 0.0])
            jr = output.get("j_right", [0.0, 0.0])
            if hasattr(jl, 'ndim') and jl.ndim == 2:
                jl = jl[0]
            if hasattr(jr, 'ndim') and jr.ndim == 2:
                jr = jr[0]
            vec[0] = float(jl[0])   # LX
            vec[1] = float(jl[1])   # LY
            vec[2] = float(jr[0])   # RX
            vec[3] = float(jr[1])   # RY
            buttons = output.get("buttons", [])
            if hasattr(buttons, 'ndim') and buttons.ndim == 2:
                buttons = buttons[0]
            for i, v in enumerate(buttons[:16]):
                vec[4 + i] = float(v)
            return vec

        raise TypeError(f"Unsupported output type: {type(output)}")
