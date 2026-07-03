from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import psutil

psutil.cpu_percent(interval=None)  # first call always returns 0.0, discard it


def _cpu_temp() -> Optional[float]:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read()) / 1000.0, 1)
    except OSError:
        pass
    try:
        temps = psutil.sensors_temperatures()  # type: ignore[attr-defined]
        for key in ("coretemp", "cpu_thermal", "k10temp", "cpu-thermal", "acpitz"):
            if key in temps and temps[key]:
                return round(temps[key][0].current, 1)
    except AttributeError:
        pass
    return None


def _nvidia_stats() -> Dict[str, Optional[float]]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        if not out:
            return {}
        parts = [p.strip() for p in out.split(",")]
        if len(parts) < 4:
            return {}
        return {
            "gpu_temp_c": float(parts[0]),
            "gpu_power_w": float(parts[1]),
            "gpu_mem_mb": float(parts[2]),
            "gpu_util_pct": float(parts[3]),
        }
    except Exception:
        return {}


def _sample() -> Dict[str, Any]:
    cpu_pct = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    gpu = _nvidia_stats()
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "cpu_pct": round(cpu_pct, 1),
        "mem_pct": round(mem.percent, 1),
        "cpu_temp_c": _cpu_temp(),
        "gpu_temp_c": gpu.get("gpu_temp_c"),
        "gpu_power_w": gpu.get("gpu_power_w"),
        "gpu_mem_mb": gpu.get("gpu_mem_mb"),
        "gpu_util_pct": gpu.get("gpu_util_pct"),
    }


class HardwareMonitor:
    def __init__(self, history_seconds: int = 120) -> None:
        self._history: deque = deque(maxlen=history_seconds)
        self._lock = threading.Lock()
        self._running = False

    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            snap = _sample()
            with self._lock:
                self._history.append(snap)
            time.sleep(1)

    def current(self) -> Dict[str, Any]:
        with self._lock:
            if self._history:
                return dict(self._history[-1])
        return _sample()

    def history(self, seconds: int = 120) -> list:
        with self._lock:
            data = list(self._history)
        return data[-seconds:] if seconds < len(data) else data


hw_monitor = HardwareMonitor()
