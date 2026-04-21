"""
Touch action overlay visualization.
Displays tap/swipe positions on the frame when viewing via VNC.
"""
import cv2
import numpy as np

# colors (BGR)
COLOR_TAP        = (0,   60, 255)   # red
COLOR_LONG_PRESS = (0,  165, 255)   # orange
COLOR_SWIPE      = (0,  200,  80)   # green
COLOR_TEXT       = (255, 255, 255)  # white


def draw_action(frame: np.ndarray, action: dict) -> np.ndarray:
    """
    action: result of ADBAction.to_dict()
    returns: copy with overlay drawn (original is not modified)
    """
    out = frame.copy()
    h, w = out.shape[:2]
    t = action.get("type", "noop")

    if t == "tap":
        px = int(action["x"] * w)
        py = int(action["y"] * h)
        _draw_tap(out, px, py, COLOR_TAP, label="TAP")

    elif t == "long_press":
        px = int(action["x"] * w)
        py = int(action["y"] * h)
        _draw_tap(out, px, py, COLOR_LONG_PRESS, label="HOLD", radius=30)

    elif t == "swipe":
        x1 = int(action["x1"] * w);  y1 = int(action["y1"] * h)
        x2 = int(action["x2"] * w);  y2 = int(action["y2"] * h)
        _draw_swipe(out, x1, y1, x2, y2, COLOR_SWIPE)

    # action type text in upper-right corner
    label = t.upper() if t != "noop" else ""
    if label:
        cv2.putText(out, label, (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2, cv2.LINE_AA)

    return out


def draw_telemetry(frame: np.ndarray, telemetry: dict) -> np.ndarray:
    """Overlay thermal/performance readings in the upper-left corner."""
    out = frame.copy()
    lines = [
        f"CPU {telemetry.get('cpu_temp', 0):.1f}C",
        f"BAT {telemetry.get('battery_temp', 0):.1f}C",
        f"GPU {telemetry.get('gpu_load_pct', 0):.0f}%",
    ]
    if telemetry.get("is_dummy"):
        lines.append("[DUMMY]")

    for i, text in enumerate(lines):
        cv2.putText(out, text, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv2.LINE_AA)
    return out


# ── Internal Drawing Helpers ──────────────────────────────────

def _draw_tap(img, px, py, color, label="", radius=36):
    cv2.circle(img, (px, py), radius, color, 4)
    cv2.circle(img, (px, py), radius - 10, color, 2)
    cv2.circle(img, (px, py), 8, color, -1)
    if label:
        cv2.putText(img, label, (px + radius + 6, py + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def _draw_swipe(img, x1, y1, x2, y2, color):
    cv2.arrowedLine(img, (x1, y1), (x2, y2), color, 4, tipLength=0.2)
    cv2.circle(img, (x1, y1), 12, color, -1)
