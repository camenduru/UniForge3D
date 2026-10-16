"""Device-memory cleanup shared by the sequential inference stages."""

import gc
import subprocess

import torch


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vram_diagnostics() -> str:
    """One-line VRAM snapshot for out-of-memory errors (best effort)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        used, free, total = (int(float(x)) for x in out.stdout.strip().split(","))
        holders = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        names = []
        for line in holders.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and int(float(parts[2])) > 100:
                names.append(f"{parts[1].split('/')[-1]}(pid {parts[0]})")
        detail = f" holders: {', '.join(names)}" if names else ""
        return (f"VRAM used {used} MiB / free {free} MiB / total {total} MiB."
                f"{detail}")
    except Exception:
        return "VRAM snapshot unavailable."
