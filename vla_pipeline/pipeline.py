"""
Main pipeline loop. Run directly on the server.

Usage:
  python pipeline.py --dummy                         # dummy mode (without NitroGen)
  python pipeline.py --device emulator-5554          # specify a device
  python pipeline.py --dummy --no-record             # run without saving
  python pipeline.py --dummy --step-interval 0.3    # adjust step interval
"""
import argparse, signal, sys, time

from adb_env      import ADBEnv
from telemetry    import TelemetryCollector
from action_mapper import ActionMapper
from nitrogen_client import build_client
from recorder     import SessionRecorder
from visualizer   import draw_action, draw_telemetry


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device",        default=None,   help="ADB serial (default: auto-detect)")
    p.add_argument("--dummy",         action="store_true", help="NitroGen dummy mode")
    p.add_argument("--no-record",     action="store_true", help="Do not save session")
    p.add_argument("--step-interval", type=float, default=0.5, help="Step interval (seconds)")
    p.add_argument("--output-dir",    default="output", help="Session save directory")
    p.add_argument("--nitrogen-host", default="localhost")
    p.add_argument("--nitrogen-port", type=int, default=5556)
    return p.parse_args()


def auto_detect_device() -> str | None:
    devices = ADBEnv.list_devices()
    if not devices:
        print("[Pipeline] No ADB devices found. Check that the emulator is running.")
        sys.exit(1)
    if len(devices) > 1:
        print(f"[Pipeline] Multiple devices detected: {devices}")
        print(f"[Pipeline] Using first device: {devices[0]}")
    return devices[0]


def main():
    args = parse_args()

    # device detection
    device = args.device or auto_detect_device()
    print(f"[Pipeline] Device: {device}")

    # module initialization
    env      = ADBEnv(device)
    mapper   = ActionMapper()
    nitrogen = build_client(
        dummy=args.dummy,
        host=args.nitrogen_host,
        port=args.nitrogen_port,
    )
    telemetry = TelemetryCollector(device_serial=device, interval_ms=500)
    recorder  = SessionRecorder(args.output_dir) if not args.no_record else None

    env.wait_for_device()
    w, h = env.get_screen_size()
    print(f"[Pipeline] Screen size: {w}x{h}")

    telemetry.start()

    # Ctrl+C handler
    running = True
    def _stop(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _stop)

    step = 0
    print("[Pipeline] Started. Press Ctrl+C to stop.")
    print("-" * 50)

    try:
        while running:
            t0 = time.time()

            # 1. screen capture
            frame = env.capture_screen()

            # 2. NitroGen inference
            nitrogen_raw = nitrogen.infer(frame)

            # 3. action mapping
            adb_action = mapper.map(nitrogen_raw)
            action_dict = adb_action.to_dict()

            # 4. execute touch
            env.execute(action_dict)

            # 5. telemetry
            tele = telemetry.get_latest()

            # 6. visualization overlay
            viz = draw_action(frame, action_dict)
            viz = draw_telemetry(viz, tele)

            # 7. record
            if recorder:
                recorder.record(viz, action_dict, nitrogen_raw, tele)

            # console status output (every 5 steps)
            step += 1
            if step % 5 == 0:
                elapsed = time.time() - t0
                fps = 1.0 / max(elapsed, 1e-6)
                print(
                    f"  step={step:5d} | {action_dict['type']:10s} | "
                    f"cpu={tele.get('cpu_temp',0):.1f}C | "
                    f"fps={fps:.1f}"
                )

            # maintain step interval
            elapsed = time.time() - t0
            sleep = args.step_interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    finally:
        telemetry.stop()
        nitrogen.close()
        if recorder:
            recorder.close()
        print("\n[Pipeline] Shutdown complete.")


if __name__ == "__main__":
    main()
