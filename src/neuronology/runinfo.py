"""Run provenance and plumbing shared by every script.

- `load_config`: YAML -> dict, plus a stable hash of its content.
- `provenance`: git commit, config hash, versions, device; stamped into every result file.
- `artifacts_dir`: root for large, uncommitted outputs (corpus, census arrays, checkpoints).
  `$NEURONOLOGY_ARTIFACTS` if set (on a pod: `$VOLUME/artifacts`), else `<repo>/artifacts`.
- `Progress`: writes `$STATUS_DIR/progress.json`, which runpod/watchdog.sh merges into its
  heartbeat `status.json`.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    cfg["_config_hash"] = config_hash(cfg)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def git_commit() -> str:
    env = os.environ.get("GIT_COMMIT")
    if env:
        return env
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "diff", "--quiet", "HEAD", "--", "src", "scripts"],
            stderr=subprocess.DEVNULL,
        ).returncode
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def provenance(cfg: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    import torch
    import transformers

    out: dict[str, Any] = {
        "git_commit": git_commit(),
        "date_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "run_id": os.environ.get("RUN_ID"),
        "pod_id": os.environ.get("RUNPOD_POD_ID"),
    }
    if cfg is not None:
        out["config_path"] = cfg.get("_config_path")
        out["config_hash"] = cfg.get("_config_hash")
        out["config"] = {k: v for k, v in cfg.items() if not k.startswith("_")}
    out.update(extra)
    return out


def artifacts_dir(*parts: str) -> Path:
    root = Path(os.environ.get("NEURONOLOGY_ARTIFACTS", REPO_ROOT / "artifacts"))
    p = root.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def results_dir(*parts: str) -> Path:
    p = REPO_ROOT.joinpath("results", *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
        f.write("\n")
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def torch_dtype(name: str):
    import torch

    table = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if name not in table:
        raise ValueError(f"dtype must be one of {sorted(table)}, got {name!r}")
    return table[name]


def sync(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def peak_memory_gb(device) -> float | None:
    import torch

    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    return None


class Progress:
    """Throttled writer of `progress.json` for the watchdog heartbeat (no-op without a dir)."""

    def __init__(self, phase: str, total: int | None, every_s: float = 20.0):
        d = os.environ.get("STATUS_DIR")
        self.path = Path(d) / "progress.json" if d else None
        self.phase, self.total, self.every_s = phase, total, every_s
        self.t0 = time.time()
        self._last = 0.0

    def update(self, done: int, force: bool = False, **fields: Any) -> dict[str, Any]:
        now = time.time()
        rate = done / max(now - self.t0, 1e-9)
        eta = (self.total - done) / rate if self.total and rate > 0 else None
        rec = {
            "phase": self.phase,
            "done": done,
            "total": self.total,
            "rate_per_s": round(rate, 2),
            "eta_s": round(eta) if eta is not None else None,
            "updated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **fields,
        }
        if self.path and (force or now - self._last >= self.every_s):
            write_json(self.path, rec)
            self._last = now
        return rec
