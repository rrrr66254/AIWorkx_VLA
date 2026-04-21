"""
메인 파이프라인 루프. 서버에서 직접 실행.

사용법:
  python pipeline.py --dummy                         # 더미 모드 (NitroGen 없이)
  python pipeline.py --device emulator-5554          # 특정 기기 지정
  python pipeline.py --dummy --no-record             # 저장 없이 실행
  python pipeline.py --dummy --step-interval 0.3    # 스텝 간격 조정
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
    p.add_argument("--device",        default=None,   help="ADB 시리얼 (기본: 자동 감지)")
    p.add_argument("--dummy",         action="store_true", help="NitroGen 더미 모드")
    p.add_argument("--no-record",     action="store_true", help="세션 저장 안 함")
    p.add_argument("--step-interval", type=float, default=0.5, help="스텝 간격 (초)")
    p.add_argument("--output-dir",    default="output", help="세션 저장 디렉토리")
    p.add_argument("--nitrogen-host", default="localhost")
    p.add_argument("--nitrogen-port", type=int, default=5556)
    return p.parse_args()


def auto_detect_device() -> str | None:
    devices = ADBEnv.list_devices()
    if not devices:
        print("[Pipeline] ADB 기기를 찾을 수 없습니다. 에뮬레이터가 실행 중인지 확인하세요.")
        sys.exit(1)
    if len(devices) > 1:
        print(f"[Pipeline] 여러 기기 감지됨: {devices}")
        print(f"[Pipeline] 첫 번째 기기 사용: {devices[0]}")
    return devices[0]


def main():
    args = parse_args()

    # 기기 감지
    device = args.device or auto_detect_device()
    print(f"[Pipeline] 기기: {device}")

    # 모듈 초기화
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
    print(f"[Pipeline] 화면 크기: {w}x{h}")

    telemetry.start()

    # Ctrl+C 처리
    running = True
    def _stop(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _stop)

    step = 0
    print("[Pipeline] 시작. Ctrl+C로 종료.")
    print("-" * 50)

    try:
        while running:
            t0 = time.time()

            # 1. 화면 캡처
            frame = env.capture_screen()

            # 2. NitroGen 추론
            nitrogen_raw = nitrogen.infer(frame)

            # 3. 액션 매핑
            adb_action = mapper.map(nitrogen_raw)
            action_dict = adb_action.to_dict()

            # 4. 터치 실행
            env.execute(action_dict)

            # 5. 텔레메트리
            tele = telemetry.get_latest()

            # 6. 시각화 오버레이
            viz = draw_action(frame, action_dict)
            viz = draw_telemetry(viz, tele)

            # 7. 기록
            if recorder:
                recorder.record(viz, action_dict, nitrogen_raw, tele)

            # 콘솔 상태 출력 (5스텝마다)
            step += 1
            if step % 5 == 0:
                elapsed = time.time() - t0
                fps = 1.0 / max(elapsed, 1e-6)
                print(
                    f"  step={step:5d} | {action_dict['type']:10s} | "
                    f"cpu={tele.get('cpu_temp',0):.1f}°C | "
                    f"fps={fps:.1f}"
                )

            # 스텝 간격 유지
            elapsed = time.time() - t0
            sleep = args.step_interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    finally:
        telemetry.stop()
        nitrogen.close()
        if recorder:
            recorder.close()
        print("\n[Pipeline] 종료 완료.")


if __name__ == "__main__":
    main()
