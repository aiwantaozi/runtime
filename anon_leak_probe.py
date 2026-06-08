#!/usr/bin/env python3
"""
Isolate which native call in the AMD detect path leaks anonymous memory.

Run each mode for a few thousand iterations and watch the Anonymous line of
/proc/self/smaps_rollup. The mode whose anon RSS grows linearly is the leak.

Usage (inside the worker container, where gpustack_runtime is installed):
    python3 anon_leak_probe.py detect   --iters 3000
    python3 anon_leak_probe.py hsa      --iters 3000
    python3 anon_leak_probe.py amdgpu   --iters 3000 --card 1
    python3 anon_leak_probe.py amdsmi   --iters 3000

`--card` is the /dev/dri/cardN index (the AMD card). For the amdgpu mode this
FORCES the libdrm init/deinit path regardless of whether detect() would enter it.
"""
from __future__ import annotations

import argparse
import gc
import sys


def anon_kb() -> int:
    with open("/proc/self/smaps_rollup") as f:
        for line in f:
            if line.startswith("Anonymous:"):
                return int(line.split()[1])
    return -1


def open_fds() -> int:
    import os

    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return -1


def run(mode: str, iters: int, card: int) -> None:
    from gpustack_runtime.detector import pyamdgpu, pyamdsmi, pyhsa

    def one_iter():
        if mode == "detect":
            from gpustack_runtime.detector import detect_devices

            detect_devices(fast=False)
        elif mode == "hsa":
            pyhsa.get_agents()
        elif mode == "amdsmi":
            pyamdsmi.amdsmi_init()
            devs = pyamdsmi.amdsmi_get_processor_handles()
            for d in devs:
                pyamdsmi.amdsmi_get_gpu_asic_info(d)
                pyamdsmi.amdsmi_get_gpu_vram_usage(d)
        elif mode == "amdgpu":
            import contextlib

            with contextlib.suppress(pyamdgpu.AMDGPUError), pyamdgpu.amdgpu_device(
                card
            ) as dev:
                pyamdgpu.amdgpu_query_gpu_info(dev)
        else:
            raise SystemExit(f"unknown mode: {mode}")

    # warm up (one-time init allocations should not count as a leak)
    one_iter()
    gc.collect()
    base_anon, base_fd = anon_kb(), open_fds()
    print(f"[{mode}] baseline anon={base_anon} KB fd={base_fd}")

    step = max(1, iters // 10)
    for i in range(1, iters + 1):
        one_iter()
        if i % step == 0:
            gc.collect()
            a, fd = anon_kb(), open_fds()
            print(
                f"[{mode}] iter={i:6d} anon={a} KB (+{a - base_anon}) "
                f"fd={fd} per_iter={(a - base_anon) / i:.1f} KB"
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["detect", "hsa", "amdsmi", "amdgpu"])
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--card", type=int, default=1)
    args = p.parse_args()
    run(args.mode, args.iters, args.card)


if __name__ == "__main__":
    sys.exit(main())
