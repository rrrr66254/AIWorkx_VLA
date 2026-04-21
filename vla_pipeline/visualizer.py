"""
터치 액션 오버레이 시각화.
VNC로 볼 때 어느 위치를 탭/스와이프하는지 프레임 위에 표시.
"""
import cv2
import numpy as np

# 색상 (BGR)
COLOR_TAP        = (0,   60, 255)   # 빨강
COLOR_LONG_PRESS = (0,  165, 255)   # 주황
COLOR_SWIPE      = (0,  200,  80)   # 초록
COLOR_TEXT       = (255, 255, 255)  # 흰색


def draw_action(frame: np.ndarray, action: dict) -> np.ndarray:
    """
    action: ADBAction.to_dict() 결과
    반환: 오버레이가 그려진 복사본 (원본 수정 안 함)
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

    # 우상단에 액션 타입 텍스트
    label = t.upper() if t != "noop" else ""
    if label:
        cv2.putText(out, label, (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2, cv2.LINE_AA)

    return out


def draw_telemetry(frame: np.ndarray, telemetry: dict) -> np.ndarray:
    """좌상단에 발열/성능 수치 오버레이."""
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


# ── 내부 드로잉 헬퍼 ─────────────────────────────────────────

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
