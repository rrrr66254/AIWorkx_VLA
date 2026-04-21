"""
NitroGen-guided RL pipeline.

When NitroGen watches the game screen and outputs a gamepad signal:
  - Exploration phase (epsilon probability): execute NitroGen action directly  <- teacher role
  - Exploitation phase (1-epsilon probability): execute action learned by RL   <- training result

Training progress:
  epsilon = 1.0  -> 100% NitroGen (early: collecting NitroGen play data)
  epsilon = 0.5  -> 50% NitroGen + 50% RL
  epsilon = 0.1  -> 10% NitroGen + 90% RL (late: RL surpasses NitroGen)

Usage:
  python pipeline_ng_rl.py --device emulator-5554
  python pipeline_ng_rl.py --no-record
  python pipeline_ng_rl.py --no-scrcpy   # disable scrcpy (ADB fallback)
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


# ── Game State Detection ────────────────────────────────────────────
DETECT_PX_X,   DETECT_PX_Y   = 800, 2104
PLAY_TAP_X,    PLAY_TAP_Y    = 810, 2220
POPUP_CLOSE_X, POPUP_CLOSE_Y = 820, 175   # result screen small X button
IAP_CLOSE_X,   IAP_CLOSE_Y   = 996, 487   # IAP popup (Permanent score boost etc.) X button
CONTINUE_PX_X, CONTINUE_PX_Y = 540, 860

GAME_PKG   = "com.kiloo.subwaysurf"
RL_CKPT    = os.path.join(os.path.expanduser("~"), "vla_pipeline", "rl_ng_checkpoint.pt")
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
    """If game is not in foreground, relaunch it. Returns True if relaunched."""
    try:
        output = env._run(["shell", "dumpsys", "activity", "activities"], timeout=5)
        if GAME_PKG not in output:
            print("[NG-RL] Game not in foreground -> relaunching")
            env._run(["shell", "monkey", "-p", GAME_PKG,
                      "-c", "android.intent.category.LAUNCHER", "1"], timeout=5)
            time.sleep(3.0)
            # close popups then PLAY
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
        print(f"[NG-RL] ensure_game_foreground error: {e}")
    return False


def handle_game_over(env: ADBEnv):
    """Game over screen -> close popups -> restart PLAY.

    Subway Surfers often chains multiple IAP popups (Permanent score boost,
    Daily reward, Mission board, etc.) so we tap the IAP close position
    several times with short pauses to dismiss all of them.
    """
    print("[NG-RL] Game over -> restart")
    # 1) IAP popup X button — tap up to 4 times to chain-close popups
    for _ in range(4):
        env._run(["shell", "input", "tap", str(IAP_CLOSE_X), str(IAP_CLOSE_Y)])
        time.sleep(0.4)
    # 2) Result screen X button (score summary screen)
    env._run(["shell", "input", "tap", str(POPUP_CLOSE_X), str(POPUP_CLOSE_Y)])
    time.sleep(0.5)
    # 3) One more IAP close in case a new popup appeared after result screen
    env._run(["shell", "input", "tap", str(IAP_CLOSE_X), str(IAP_CLOSE_Y)])
    time.sleep(0.4)
    # 4) PLAY button (back on main menu now)
    env._run(["shell", "input", "tap", str(PLAY_TAP_X), str(PLAY_TAP_Y)])
    time.sleep(0.8)
    # 5) Game start confirmation tap (character selection / mission screen)
    env._run(["shell", "input", "tap", "540", "1200"])
    time.sleep(0.8)


def action_dict_to_idx(d: dict) -> int:
    """ADB action dict -> index 0~4 conversion."""
    if d.get("type") == "noop":
        return 0
    if d.get("type") == "swipe":
        dx = d["x2"] - d["x1"]
        dy = d["y2"] - d["y1"]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0   # zero-length swipe (cursor clamped at edge) -> noop
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
    p.add_argument("--no-scrcpy",     action="store_true", help="disable scrcpy, use ADB screencap")
    return p.parse_args()


def auto_detect_device() -> str:
    devices = ADBEnv.list_devices()
    if not devices:
        print("[NG-RL] No ADB device found.")
        sys.exit(1)
    return devices[0]


def main():
    args   = parse_args()
    device = args.device or auto_detect_device()
    print(f"[NG-RL] Device: {device}  |  NitroGen-guided Double DQN")

    env      = ADBEnv(device)
    mapper   = ActionMapper()

    # ── scrcpy high-speed capture ─────────────────────────────────
    capture = None
    if not args.no_scrcpy:
        try:
            capture = ScrcpyCapture(device=device, max_fps=30)
            capture.start(timeout=20)
        except Exception as e:
            print(f"[NG-RL] scrcpy failed ({e}), falling back to ADB screencap")
            capture = None

    SCRCPY_MAX_FRAME_AGE = 2.0   # seconds; fall back to ADB if no new frame

    def get_frame():
        if capture is not None:
            age = time.time() - capture.last_frame_time
            if age < SCRCPY_MAX_FRAME_AGE:
                f = capture.get_frame()
                return f if f is not None else env.capture_screen()
        # scrcpy stalled or not running -> use ADB screencap (always fresh)
        return env.capture_screen()

    nitrogen = build_client(
        dummy=args.dummy,
        host=args.nitrogen_host,
        port=args.nitrogen_port,
    )
    telemetry = TelemetryCollector(device_serial=device, interval_ms=500)
    recorder  = SessionRecorder(args.output_dir) if not args.no_record else None

    # ── RL agent (FC, 23D state) ──────────────────────────────────
    agent = DQNAgent(
        device          = "cuda",
        epsilon_start   = 1.0,
        epsilon_end     = 0.10,    # keep at least 10% NitroGen exploration
        epsilon_decay   = 0.998,   # reach epsilon=0.1: ~3,000 train steps ~ 3-4 hours
    )
    if os.path.exists(RL_CKPT):
        agent.load(RL_CKPT)
        print(f"[NG-RL] Checkpoint loaded: epsilon={agent.epsilon:.3f}")
    else:
        print(f"[NG-RL] Starting new agent (epsilon={agent.epsilon:.2f})")
        print("[NG-RL] NitroGen plays 100% in the early stage.")

    env.wait_for_device()
    w, h = env.get_screen_size()
    print(f"[NG-RL] Screen size: {w}x{h}")
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
    ng_count        = 0   # cumulative NitroGen actions since pipeline start
    rl_count        = 0   # cumulative RL actions since pipeline start
    detect_frame    = None   # dedicated ADB frame for game-state detection

    print("[NG-RL] Starting. Press Ctrl+C to stop.")
    print("-" * 60)

    try:
        while running:
            t0 = time.time()

            # 1. Capture action frame (scrcpy or ADB)
            frame = get_frame()

            # 2. Detection frame: dedicated ADB screencap every 5 steps for
            #    guaranteed freshness. scrcpy can lag behind the real screen state
            #    (e.g. game-over pixel missed because scrcpy shows cached frame).
            if step % 5 == 0 or detect_frame is None:
                detect_frame = env.capture_screen()

            # ensure_game_foreground is an ADB call, keep throttled to every 30 steps
            if step % 30 == 0:
                ensure_game_foreground(env)

            # Check game_over FIRST: the result screen can trigger both game_over
            # and continue_dialog pixels simultaneously (blue revive button at 540,860).
            # Handling game_over takes priority.
            if detect_game_over(detect_frame):
                if prev_state is not None:
                    ns = DQNAgent.zero_state()
                    agent.store(prev_state, prev_action, -1.0, ns, True)
                    prev_state = prev_action = None
                game_over_count += 1
                handle_game_over(env)
                detect_frame = None   # force ADB refresh after restart
                step += 1
                continue

            # Continue dialog ("Continue for coins?") — only reached when NOT game over
            if detect_continue_dialog(detect_frame):
                if prev_state is not None:
                    ns = DQNAgent.zero_state()
                    agent.store(prev_state, prev_action, -1.0, ns, True)
                    prev_state = prev_action = None
                # tap "No" side or wait for auto-dismiss
                env._run(["shell", "input", "tap", "820", "1700"])  # right side = No/close
                step += 1
                time.sleep(0.5)
                continue

            # 3. NitroGen inference (always runs - state extraction + exploration policy)
            nitrogen_raw = nitrogen.infer(frame)
            state        = agent.extract_state(nitrogen_raw)

            # 4. Store previous transition (survival reward +0.1)
            if prev_state is not None:
                agent.store(prev_state, prev_action, +0.1, state, False)

            # 5. Action selection: epsilon -> NitroGen,  1-epsilon -> RL
            import random
            if random.random() < agent.epsilon:
                # exploration: use NitroGen action
                ng_adb    = mapper.map(nitrogen_raw)
                action_dict = ng_adb.to_dict()
                action_idx  = action_dict_to_idx(action_dict)
                ng_count += 1
                src = "NG"
            else:
                # exploitation: RL learned action
                action_idx  = agent.select_action(state)
                action_dict = agent.get_action_dict(action_idx)
                rl_count += 1
                src = "RL"

            prev_state  = state
            prev_action = action_idx
            agent.total_steps += 1

            # 6. Train (every 4 steps)
            if step % 4 == 0:
                rl_loss = agent.train()

            # 7. Save checkpoint
            if step % SAVE_EVERY == 0 and step > 0:
                agent.save(RL_CKPT)
                total_acts = ng_count + rl_count
                ng_pct = 100 * ng_count / max(total_acts, 1)
                print(f"[NG-RL] Saved: epsilon={agent.epsilon:.3f}  "
                      f"env_steps={agent.total_steps}  "
                      f"best={agent.best_episode_reward:.2f}  "
                      f"NG:{ng_pct:.0f}% RL:{100-ng_pct:.0f}%")

            # 8. Execute action
            env.execute(action_dict)

            # 9. Telemetry + recording
            tele = telemetry.get_latest()
            if recorder:
                viz = draw_action(frame, action_dict)
                viz = draw_telemetry(viz, tele)
                recorder.record(viz, action_dict, nitrogen_raw, tele)

            # 10. Console output (every 5 steps)
            step += 1
            if step % 5 == 0:
                elapsed    = time.time() - t0
                aname      = ACTION_NAMES[prev_action] if prev_action is not None else "?"
                loss_str   = f"{rl_loss:.4f}" if rl_loss is not None else "  --  "
                total_acts = ng_count + rl_count
                ng_pct     = 100 * ng_count / max(total_acts, 1)   # cumulative since start
                scrcpy_fps = f"{capture.current_fps:.0f}" if capture else "ADB"
                print(
                    f"  step={step:5d} | {src}:{aname:5s} | "
                    f"e={agent.epsilon:.3f} | loss={loss_str} | "
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
        print(f"\n[NG-RL] Final save: epsilon={agent.epsilon:.3f}  "
              f"env_steps={agent.total_steps}  "
              f"episodes={agent.episode_count}  "
              f"best={agent.best_episode_reward:.2f}")
        print(f"[NG-RL] Action ratio: NitroGen {ng_pct:.0f}%  /  RL {100-ng_pct:.0f}%")
        telemetry.stop()
        nitrogen.close()
        if capture:
            capture.stop()
        if recorder:
            recorder.close()
        print("\n[NG-RL] Shutdown complete.")


if __name__ == "__main__":
    main()
