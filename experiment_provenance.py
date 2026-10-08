"""Lightweight code, data, settings and environment records for training runs."""

import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def collect_provenance(data_root, loader_module, checkpoints, args):
    root = Path(__file__).resolve().parent
    data_root = Path(data_root).resolve()

    def git(*command):
        try:
            proc = subprocess.run(["git", *command], cwd=root, capture_output=True,
                                  text=True, timeout=10)
            return proc.stdout.strip() if proc.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    device = torch.device(args.device)
    return {
        "args": vars(args).copy(),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "numpy": np.__version__,
                        "cuda_runtime": torch.version.cuda,
                        "device": str(device),
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "data_root": str(data_root),
        "data_metadata": str(data_root / "metadata.json"),
        "actual_loader": str(Path(loader_module.__file__).resolve()),
        "checkpoints": {key: str(Path(path).resolve()) for key, path in checkpoints.items()},
    }
