"""Small, read-only fingerprints for reproducible local/server experiments."""

import hashlib
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def first_stage_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not name.startswith("W"):
            value = value.detach().cpu().contiguous()
            digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}".encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def collect_provenance(data_root, loader_module, checkpoints, args):
    root = Path(__file__).resolve().parent
    data_root = Path(data_root).resolve()
    files = [data_root / "metadata.json", Path(loader_module.__file__)]
    for split in ("train", "val"):
        candidates = (data_root / f"{split}_index.csv", data_root / split / "index.csv")
        index = next((p for p in candidates if p.exists()), None)
        if index is not None:
            files.append(index)
    code = [root / name for name in (
        "train.py", "evaluate.py", "metrics.py", "boundary.py", "radiounet.py",
        "lunar_dataset.py", "RadioUNet/modules.py", "experiment_provenance.py",
        "ablate_boundary.py")]

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
        "code": [fingerprint(p) for p in code if p.exists()],
        "data_metadata_and_actual_loader": [fingerprint(p) for p in files],
        "checkpoints": {key: fingerprint(path) for key, path in checkpoints.items()},
        "data_array_hashes": "Not computed: large arrays are identified by path/version supplied by the user; metadata/index hashes alone do not prove array identity.",
    }
