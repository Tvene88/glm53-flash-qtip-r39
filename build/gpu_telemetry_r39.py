#!/usr/bin/env python3
"""gpu_telemetry_r39.py - one nvidia-smi sample per second, appended and fsync'ed, so the last seconds before a hard
PC crash survive on disk. Added 2026-11-16 after the crash during the Layer-27 lab. No torch, no CUDA context.

    python gpu_telemetry_r39.py OUT.csv [interval_s]
Stops when the file OUT.csv.stop exists.
"""
import os
import subprocess
import sys
import time

FIELDS = "timestamp,power.draw,temperature.gpu,utilization.gpu,memory.used,fan.speed,clocks.sm,pstate"


def main() -> None:
    out = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    stop = out + ".stop"
    if os.path.exists(stop):
        os.remove(stop)
    new = not os.path.exists(out)
    with open(out, "a", encoding="utf-8") as fh:
        if new:
            fh.write(FIELDS + ",ram_free_gib\n")
        while not os.path.exists(stop):
            t0 = time.time()
            try:
                line = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip()
            except Exception as ex:  # noqa: BLE001
                line = f"ERR {ex!r}"
            ram = ""
            try:
                import ctypes

                class MS(ctypes.Structure):
                    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                        (n, ctypes.c_ulonglong) for n in ("tp", "ap", "tpf", "apf", "tv", "av", "aev")
                    ]

                ms = MS()
                ms.dwLength = ctypes.sizeof(MS)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
                ram = f"{ms.ap / 2**30:.1f}"
            except Exception:  # noqa: BLE001
                pass
            fh.write(f"{line},{ram}\n")
            fh.flush()
            os.fsync(fh.fileno())
            time.sleep(max(0.0, interval - (time.time() - t0)))
    os.remove(stop)


if __name__ == "__main__":
    main()
