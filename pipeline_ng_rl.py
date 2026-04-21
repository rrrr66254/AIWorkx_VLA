"""
NitroGen-guided RL 파이프라인.

NitroGen이 게임 화면을 보고 gamepad 신호를 출력하면:
  - 탐험 단계 (ε 확률): NitroGen 액션 그대로 실행  ← 교사 역할
  - 활용 단계 (1-ε 확률): RL이 학습한 액션 실행    ← 학습 결과

훈련 진행:
  ε = 1.0  → 100% NitroGen (초반: NitroGen 플레이 데이터 수집)
  ε = 0.5  → 50% NitroGen + 50% RL
  ε = 0.1  → 10% NitroGen + 90% RL (후반: RL이 NitroGen 초월)

사용법:
  python pipeline_ng_rl.py --device emulator-5554
  python pipeline_ng_rl.py --no-record
  python pipeline_ng_rl.py --no-scrcpy   # scrcpy 비활성화 (ADB 폴백)
"""
import argparse, os, signal, sys, time, subprocess
import numpy as np

from adb_env        import ADBEnv
from fast_capture   import ScrcpyCapture
from telemetry      import TelemetryCollector
from action_mapper  import ActionMapper
from nitrogen_client import build_client
from recorder       import SessionRecorder
from visualizer     import draw_action, draw_telemetry
from rl_agent       import DQNAgent, ACTION_NAMES


# ── 게임 상태 감지 ──────────────────────────────────────────────
DETECT_PX_X,   DETECT_PX_Y   = 800, 2104
PLAY_TAP_X,    PLAY_TAP_Y    = 810, 2220
POPUP_CLOSE_X, POPUP_CLOSE_Y = 820, 175   # 결과화면 작은 X 버튼
IAP_CLOSE_X,   IAP_CLOSE_Y   = 1040, 248  # IAP 팝업 (Permanent score boost 등) X 버튼
CONTINUE_PX_X, CONTINUE_PX_Y = 540, 860

GAME_PKG   = "com.kiloo.subwaysurf"
RL_CKPT    = "/home/sltrain/vla_pipeline/rl_ng_checkpoint.pt"
SAVE_EVERY = 200


def detect_game_over(frame: np.ndarray) -> bool:
    h, w = frame.shape[:2]
    py, px = min(DETECT_PX_Y, h-1), min(DETECT_PX_X, w-1)
    b, g, r = int(frame[py,px,0]), int(frame[py,px,1]), int(frame[py,px,2])
    return g > 140 and b < 80 and (g - r) > 60


def detect_continue_dialog(frame: np.ndarray) -> bool:
    h, w = frame.shape[:2]
    py, px = min(CONTINUE_PX_Y, h-1), min(CONTINUE_PX_X, w-1)
    b, r = int(frame[py,px,0]), int(frame[py,px,2])
    return b > 200 and r < 50


def ensure_game_foreground(env: ADBEnv) -> bool:
    """게임이 포그라운드가 아니면 재실행. 재실행했으면 True 반환."""
    try:
        output = env._run(["shell", "dumpsys", "activity", "activities"], timeout=5)
        if GAME_PKG not in output:
            print("[NG-RL] 게임 비포그라운드 감지 → 재실행")
            env._run(["shell", "monkey", "-p", GAME_PKG,
                      "-c", "android.intent.category.LAUNCHER", "1"], timeout=5)
            time.sleep(3.0)
            # 팝업 닫기 후 PLAY
            env._run(["shell", "input", "tap", str(IAP_CLOSE_X), str(IAP_CLOSE_Y)])
            time.sleep(0.3)
            env._run(["shell", "input", "keyevent", "KEYCODE_BACK"])
            time.sleep(0.3)
            env._run(["shell", "input", "tap", str(PLAY_TAP_X), str(PLAY_TAP_Y)])
            time.sleep(1.0)
            env._run(["shell", "input", "tap", "540", "1200"])
            time.sleep(0.8)
            return True
    except Exception as e:
        print(f"[NG-RL] ensure_game_foreground 오류: {e}")
    return False


def handle_game_over(env: ADBEnv):
    """게임 오버 화면 → 팝업 닫기 → PLAY 재시작."""
    print("[NG-RL] 게임 오버 → 재시작")
    # 1) IAP 팝업 X 버튼 (오른쪽 상단 빨간 원)
    env._run(["shell", "input", "tap", str(IAP_CLOSE_X), str(IAP_CLOSE_Y)])
    time.sleep(0.25)
    # 2) 기존 결과화면 X 버튼
    env._run(["shell", "input", "tap", str(POPUP_CLOSE_X), str(POPUP_CLOSE_Y)])
    time.sleep(0.25)
    # 3) BACK 키로 남은 팝업 닫기
    env._run(["shell", "input", "keyevent", "KEYCODE_BACK"])
    time.sleep(0.3)
    # 4) PLAY 버튼
    env._run(["shell", "input", "tap", str(PLAY_TAP_X), str(PLAY_TAP_Y)])
    time.sleep(0.8)
    # 5) 캐릭터 선택 / 게임 시작 확인 탭
    env._run(["shell", "input", "tap", "540", "1200"])
    time.sleep(0.8)


def action_dict_to_idx(d: dict) -> int:
    """ADB action dict → 0~4 인덱스 변환."""
    if d.get("type") == "noop":
        return 0
    if d.get("type") == "swipe":
        dx = d["x2"] - d["x1"]
        dy = d["y2"] - d["y1"]
        if abs(dx) >= abs(dy):
            return 1 if dx < 0 else 2   # LEFT / RIGHT
        else:
            return 3 if dy < 0 else 4   # UP / DOWN
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device",        default=None)
    p.add_argument("--no-record",     action="store_true")
    p.add_argument("--step-interval", type=float, default=0.3)
    p.add_argument("--output-dir",    default="output_ng_rl")
    p.add_argument("--nitrogen-host", default="localhost")
    p.add_argument("--nitrogen-port", type=int, default=5556)
    p.add_argument("--dummy",         action="store_true")
    p.add_argument("--no-scrcpy",     action="store_true", help="scrcpy 비활성화, ADB screencap 사용")
    return p.parse_args()


def auto_detect_device() -> str:
    devices = ADBEnv.list_devices()
    if not devices:
        print("[NG-RL] ADB 기기 없음.")
        sys.exit(1)
    return devices[0]


def main():
    args   = parse_args()
    device = args.device or auto_detect_device()
    print(f"[NG-RL] 기기: {device}  |  NitroGen-guided Double DQN")

    env      = ADBEnv(device)
    mapper   = ActionMapper()

    # ── scrcpy 고속 캡처 ──────────────────────────────────────
    capture = None
    if not args.no_scrcpy:
        try:
            capture = ScrcpyCapture(device=device, max_fps=30)
            capture.start(timeout=20)
        except Exception as e:
            print(f"[NG-RL] scrcpy 실패({e}), ADB screencap으로 폴백")
            capture = None

    def get_frame():
        if capture is not None:
            f = capture.get_frame()
            return f if f is not None else env.capture_screen()
        return env.capture_screen()

    nitrogen = build_client(
        dummy=args.dummy,
        host=args.nitrogen_host,
        port=args.nitrogen_port,
    )
    telemetry = TelemetryCollector(device_serial=device, interval_ms=500)
    recorder  = SessionRecorder(args.output_dir) if not args.no_record else None

    # ── RL 에이전트 (FC, 23D 상태) ───────────────────────────
    agent = DQNAgent(
        device          = "cuda",
        epsilon_start   = 1.0,
        epsilon_end     = 0.10,    # 최소 10%는 NitroGen 탐험 유지
        epsilon_decay   = 0.998,   # ε=0.1 도달: ~3,000 학습스텝 ≈ 3~4시간
    )
    if os.path.exists(RL_CKPT):
        agent.load(RL_CKPT)
        print(f"[NG-RL] 체크포인트 로드: ε={agent.epsilon:.3f}")
    else:
        print(f"[NG-RL] 새 에이전트 시작 (ε={agent.epsilon:.2f})")
        print("[NG-RL] 초반에는 NitroGen이 100% 플레이합니다.")

    env.wait_for_device()
    w, h = env.get_screen_size()
    print(f"[NG-RL] 화면 크기: {w}x{h}")
    telemetry.start()

    running = True
    def _stop(sig, frame_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    step            = 0
    game_over_count = 0
    prev_state      = None
    prev_action     = None
    rl_loss         = None
    ng_count        = 0   # NitroGen 액션 선택 횟수
    rl_count        = 0   # RL 액션 선택 횟수

    print("[NG-RL] 시작. Ctrl+C로 종료.")
    print("-" * 60)

    try:
        while running:
            t0 = time.time()

            # 1. 화면 캡처 (scrcpy or ADB)
            frame = get_frame()

            # 2. 게임 상태 감지 (3스텝마다)
            if step % 3 == 0:
                # 게임이 포그라운드인지 확인 (30스텝마다)
                if step % 30 == 0:
                    ensure_game_foreground(env)

                if detect_continue_dialog(frame):
                    if prev_state is not None:
                        ns = DQNAgent.zero_state()
                        agent.store(prev_state, prev_action, -1.0, ns, True)
                        prev_state = prev_action = None
                    step += 1
                    time.sleep(0.5)
                    continue

                if detect_game_over(frame):
                    if prev_state is not None:
                        ns = DQNAgent.zero_state()
                        agent.store(prev_state, prev_action, -1.0, ns, True)
                        prev_state = prev_action = None
                    game_over_count += 1
                    handle_game_over(env)
                    step += 1
                    continue

            # 3. NitroGen 추론 (항상 실행 - 상태 추출 + 탐험 정책)
            nitrogen_raw = nitrogen.infer(frame)
            state        = agent.extract_state(nitrogen_raw)

            # 4. 이전 transition 저장 (생존 보상 +0.1)
            if prev_state is not None:
                agent.store(prev_state, prev_action, +0.1, state, False)

            # 5. 액션 결정: ε → NitroGen,  1-ε → RL
            import random
            if random.random() < agent.epsilon:
                # 탐험: NitroGen 액션 사용
                ng_adb    = mapper.map(nitrogen_raw)
                action_dict = ng_adb.to_dict()
                action_idx  = action_dict_to_idx(action_dict)
                ng_count   += 1
                src = "NG"
            else:
                # 활용: RL 학습 액션
                action_idx  = agent.select_action(state)
                action_dict = agent.get_action_dict(action_idx)
                rl_count   += 1
                src = "RL"

            prev_state  = state
            prev_action = action_idx
            agent.total_steps += 1

            # 6. 학습 (4스텝마다)
            if step % 4 == 0:
                rl_loss = agent.train()

            # 7. 체크포인트 저장
            if step % SAVE_EVERY == 0 and step > 0:
                agent.save(RL_CKPT)
                total_acts = ng_count + rl_count
                ng_pct = 100 * ng_count / max(total_acts, 1)
                print(f"[NG-RL] 저장: ε={agent.epsilon:.3f}  "
                      f"env_steps={agent.total_steps}  "
                      f"best={agent.best_episode_reward:.2f}  "
                      f"NG:{ng_pct:.0f}% RL:{100-ng_pct:.0f}%")

            # 8. 액션 실행
            env.execute(action_dict)

            # 9. 텔레메트리 + 기록
            tele = telemetry.get_latest()
            if recorder:
                viz = draw_action(frame, action_dict)
                viz = draw_telemetry(viz, tele)
                recorder.record(viz, action_dict, nitrogen_raw, tele)

            # 10. 콘솔 출력 (5스텝마다)
            step += 1
            if step % 5 == 0:
                elapsed  = time.time() - t0
                fps      = 1.0 / max(elapsed, 1e-6)
                aname    = ACTION_NAMES[prev_action] if prev_action is not None else "?"
                loss_str = f"{rl_loss:.4f}" if rl_loss is not None else "  --  "
                total_acts = ng_count + rl_count
                ng_pct = 100 * ng_count / max(total_acts, 1)
                scrcpy_fps = f"{capture.current_fps:.0f}" if capture else "ADB"
                print(
                    f"  step={step:5d} | {src}:{aname:5s} | "
                    f"ε={agent.epsilon:.3f} | loss={loss_str} | "
                    f"buf={len(agent.buffer):5d} | "
                    f"NG:{ng_pct:.0f}%|RL:{100-ng_pct:.0f}% | "
                    f"cap={scrcpy_fps}fps | dies={game_over_count}"
                )

            elapsed = time.time() - t0
            sleep   = args.step_interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    finally:
        agent.save(RL_CKPT)
        total_acts = ng_count + rl_count
        ng_pct = 100 * ng_count / max(total_acts, 1)
        print(f"\n[NG-RL] 최종 저장: ε={agent.epsilon:.3f}  "
              f"env_steps={agent.total_steps}  "
              f"episodes={agent.episode_count}  "
              f"best={agent.best_episode_reward:.2f}")
        print(f"[NG-RL] 액션 비율: NitroGen {ng_pct:.0f}%  /  RL {100-ng_pct:.0f}%")
        telemetry.stop()
        nitrogen.close()
        if capture:
            capture.stop()
        if recorder:
            recorder.close()
        print("\n[NG-RL] 종료 완료.")


if __name__ == "__main__":
    main()
