"""
Thermal/performance telemetry collector — runs as a background thread.
Emulator: returns dummy values. Switches to real measurements automatically when a physical device is connected.
"""
import subprocess, threading, time, re
from dataclasses import dataclass, field, asdict


@dataclass
class TelemetrySnapshot:
    timestamp:   float = 0.0
    cpu_temp:    float = 0.0   # degrees C
    gpu_temp:    float = 0.0   # degrees C (physical device only)
    battery_temp:float = 0.0   # degrees C
    cpu_freq_mhz:float = 0.0   # MHz
    gpu_load_pct:float = 0.0   # % (physical device only)
    is_dummy:    bool  = False

    def to_dict(self) -> dict:
        return asdict(self)


class TelemetryCollector:
    def __init__(self, device_serial: str = None, interval_ms: int = 500):
        self.serial   = device_serial
        self.interval = interval_ms / 1000.0
        self._base    = ["adb"] + (["-s", device_serial] if device_serial else [])
        self._is_emu  = (device_serial is None or device_serial.startswith("emulator-"))
        self._latest: TelemetrySnapshot = TelemetrySnapshot()
        self._history: list[TelemetrySnapshot] = []
        self._running = False
        self._thread  = None

    # ── Public API ────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def get_latest(self) -> dict:
        return self._latest.to_dict()

    def get_history(self) -> list[dict]:
        return [s.to_dict() for s in self._history]

    # ── Internal Loop ─────────────────────────────────────────

    def _loop(self):
        while self._running:
            snap = self._collect_dummy() if self._is_emu else self._collect_real()
            snap.timestamp = time.time()
            self._latest   = snap
            self._history.append(snap)
            time.sleep(self.interval)

    def _shell(self, cmd: str, timeout=3) -> str:
        try:
            r = subprocess.run(
                self._base + ["shell", cmd],
                capture_output=True, timeout=timeout
            )
            return r.stdout.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    # ── Emulator Dummy ────────────────────────────────────────

    def _collect_dummy(self) -> TelemetrySnapshot:
        import random, math
        t = time.time()
        # add noise to simulate realistic readings
        return TelemetrySnapshot(
            cpu_temp     = 35.0 + 5 * math.sin(t / 30) + random.uniform(-1, 1),
            gpu_temp     = 0.0,
            battery_temp = 30.0 + random.uniform(-0.5, 0.5),
            cpu_freq_mhz = 1800 + random.uniform(-100, 100),
            gpu_load_pct = 0.0,
            is_dummy     = True,
        )

    # ── Physical Device Collection ────────────────────────────

    def _collect_real(self) -> TelemetrySnapshot:
        snap = TelemetrySnapshot(is_dummy=False)

        # CPU temperature (average across cpu-related thermal zones)
        snap.cpu_temp = self._read_cpu_temp()

        # battery temperature
        batt = self._shell("dumpsys battery | grep temperature")
        m = re.search(r"temperature:\s*(\d+)", batt)
        if m:
            snap.battery_temp = int(m.group(1)) / 10.0  # unit: 0.1 degrees C

        # CPU frequency (based on cpu0)
        freq = self._shell(
            "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
        )
        if freq.isdigit():
            snap.cpu_freq_mhz = int(freq) / 1000.0

        # GPU load (Adreno, Samsung devices)
        gpu_busy = self._shell("cat /sys/class/kgsl/kgsl-3d0/gpubusy")
        if gpu_busy:
            parts = gpu_busy.split()
            if len(parts) >= 2 and parts[1] != "0":
                snap.gpu_load_pct = int(parts[0]) / int(parts[1]) * 100

        return snap

    def _read_cpu_temp(self) -> float:
        out = self._shell("cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null")
        temps = []
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                v = int(line)
                # convert if unit is millidegrees
                temps.append(v / 1000.0 if v > 1000 else float(v))
        return sum(temps) / len(temps) if temps else 0.0
