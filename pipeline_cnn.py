"""
CNN DQN pipeline. Trains directly from game screen without NitroGen.

Game screen -> CNN 4-frame stack -> Double DQN -> ADB swipe

Usage:
  python pipeline_cnn.py --device emulator-5554
  python pipeline_cnn.py --no-record --step-interval 0.3
"""
import argparse, os, signal, sys, time
import numpy as np

from adb_env        import ADBEnv
from telemetry      import TelemetryCollector
from rl_agent_cnn   import CNNDQNAgent, ACTION_NAMES


# ── Game State Detection Pixels ───────────────────────────────────────
DETECT_PX_X,   DETECT_PX_Y   = 800, 2104   # result screen (green)
PLAY_TAP_X,    PLAY_TAP_Y    = 800, 2200
POPUP_CLOSE_X, POPUP_CLOSE_Y = 820, 175
CONTINUE_PX_X, CONTINUE_PX_Y = 540, 860    # Continue? (blue)

RL_CKPT    = os.path.join(os.path.expanduser("~"), "vla_pipeline", "rl_cnn_checkpoint.pt")
SAVE_EVERY = 200   # save checkpoint every N steps


def detect_game_over(frame: np.ndarray) -> bool:
    h, w = frame.shape[:2]
    py, px = min(DETECT_PX_Y, h-1), min(DETECT_PX_X, w-1)
    b = int(frame[py, px, 0])
    g = int(frame[py, px, 1])
    r = int(frame[py, px, 2])
    return g > 140 and b < 80 and (g - r) > 60


def detect_continue_dialog(frame: np.ndarray) -> bool:
    h, w = frame.shape[:2]
    py, px = min(CONTINUE_PX_Y, h-1), min(CONTINUE_PX_X, w-1)
    b = int(frame[py, px, 0])
    r = int(frame[py, px, 2])
    return b > 200 and r < 50


def handle_game_over(env: ADBEnv):
    """Result screen -> tap PLAY."""
    print("[CNN-Pipeline] Game over -> restarting")
    env._run(["shell", "input", "tap", str(POPUP_CLOSE_X), str(POPUP_CLOSE_Y)])
    time.sleep(0.3)
    env._run(["shell", "input", "tap", str(PLAY_TAP_X), str(PLAY_TAP_Y)])
    time.sleep(0.8)
    env._run(["shell", "input", "tap", "540", "1200"])
    time.sleep(0.8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device",        default=None)
    p.add_argument("--no-record",     action="store_true")
    p.add_argument("--step-interval", type=float, default=0.3)
    p.add_argument("--output-dir",    default="output_cnn")
    return p.parse_args()


def auto_detect_device() -> str:
    devices = ADBEnv.list_devices()
    if not devices:
        print("[CNN-Pipeline] No ADB device found. Check emulator.")
        sys.exit(1)
    return devices[0]


def main():
    args   = parse_args()
    device = args.device or auto_detect_device()
    print(f"[CNN-Pipeline] Device: {device}  |  RL: CNN Double DQN")

    env       = ADBEnv(device)
    telemetry = TelemetryCollector(device_serial=device, interval_ms=500)

    # ── CNN DQN Agent ──────────────────────────────────────────
    agent = CNNDQNAgent(device="cuda")
    if os.path.exists(RL_CKPT):
        agent.load(RL_CKPT)
    else:
        print(f"[CNN-RL] Starting new agent (epsilon={agent.epsilon:.2f})")

    env.wait_for_device()
    w, h = env.get_screen_size()
    print(f"[CNN-Pipeline] Screen size: {w}x{h}")
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

    # Initialize stack with first frame
    init_frame = env.capture_screen()
    agent.reset_stack(init_frame)

    print("[CNN-Pipeline] Started. Press Ctrl+C to stop.")
    print("-" * 60)

    try:
        while running:
            t0 = time.time()

            # 1. Capture screen
            frame = env.capture_screen()

            # 2. Detect game state (every 3 steps)
            if step % 3 == 0:
                # Continue? dialog
                if detect_continue_dialog(frame):
                    if prev_state is not None:
                        ns = agent.zero_state()
                        agent.store(prev_state, prev_action, -1.0, ns, True)
                        prev_state = prev_action = None
                    agent.reset_stack(frame)
                    step += 1
                    time.sleep(0.5)
                    continue

                # Result screen (game over)
                if detect_game_over(frame):
                    if prev_state is not None:
                        ns = agent.zero_state()
                        agent.store(prev_state, prev_action, -1.0, ns, True)
                        prev_state = prev_action = None
                    game_over_count += 1
                    handle_game_over(env)
                    # New episode start -> reset stack
                    time.sleep(0.3)
                    new_frame = env.capture_screen()
                    agent.reset_stack(new_frame)
                    step += 1
                    continue

            # 3. Extract state (frame stack)
            state = agent.get_state(frame)

            # 4. Store previous transition (survival reward +0.1)
            if prev_state is not None:
                agent.store(prev_state, prev_action, +0.1, state, False)

            # 5. Select action (epsilon-greedy)
            action_idx  = agent.select_action(state)
            action_dict = agent.get_action_dict(action_idx)
            prev_state  = state
            prev_action = action_idx
            agent.total_steps += 1

            # 6. Train (every 4 steps)
            if step % 4 == 0:
                rl_loss = agent.train()

            # 7. Save checkpoint
            if step % SAVE_EVERY == 0 and step > 0:
                agent.save(RL_CKPT)
                print(f"[CNN-RL] Saved: epsilon={agent.epsilon:.4f}  "
                      f"env_steps={agent.total_steps}  "
                      f"train_steps={agent.train_steps}  "
                      f"best_reward={agent.best_episode_reward:.2f}  "
                      f"episodes={agent.episode_count}")

            # 8. Execute action
            env.execute(action_dict)

            # 9. Console output (every 5 steps)
            step += 1
            if step % 5 == 0:
                elapsed  = time.time() - t0
                fps      = 1.0 / max(elapsed, 1e-6)
                aname    = ACTION_NAMES[prev_action] if prev_action is not None else "?"
                loss_str = f"{rl_loss:.4f}" if rl_loss is not None else "  --  "
                tele     = telemetry.get_latest()
                print(
                    f"  step={step:5d} | {aname:5s} | "
                    f"epsilon={agent.epsilon:.4f} | loss={loss_str} | "
                    f"buf={len(agent.buffer):5d} | "
                    f"cpu={tele.get('cpu_temp', 0):.1f}C | "
                    f"fps={fps:.1f} | dies={game_over_count}"
                )

            elapsed = time.time() - t0
            sleep   = args.step_interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    finally:
        agent.save(RL_CKPT)
        print(f"\n[CNN-RL] Final save: epsilon={agent.epsilon:.4f}  "
              f"env_steps={agent.total_steps}  "
              f"train_steps={agent.train_steps}  "
              f"episodes={agent.episode_count}  "
              f"best_reward={agent.best_episode_reward:.2f}")
        telemetry.stop()
        print("\n[CNN-Pipeline] Shutdown complete.")


if __name__ == "__main__":
    main()
