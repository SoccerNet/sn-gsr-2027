"""Shared helpers: configuration loading, per-sequence paths and CLI arguments.

Every pipeline step is run as ``python -m sngsr.<step> --config <yaml> --sequence <name>``.
Frames are read from ``DATA_DIR/<SPLIT>/<sequence>/img1`` and are never modified;
every intermediate and final file is written to ``OUTPUT_DIR/<SPLIT>/<sequence>/``.
"""

import argparse
import os
from dataclasses import dataclass

import yaml

# Root of the baseline (the directory that contains configs/, checkpoints/, assets/).
BASELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Column layout of every image-space tracking file written by the pipeline.
TRACK_COLUMNS = ["frame", "track_id", "x", "y", "w", "h", "score", "role", "jersey", "color", "team"]


def resolve(path: str) -> str:
    """Resolve a path from the config relative to the baseline directory."""
    if path is None or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(BASELINE_DIR, path))


def apply_overrides(cfg: dict, overrides) -> dict:
    """Apply ``KEY=VALUE`` overrides; nested keys use dots (``QWEN.MODEL_ID=...``).

    Values are parsed as YAML, so ``QWEN.ENABLED=false`` gives a boolean and
    ``TRACKER.BATCH_SIZE=8`` an integer.
    """
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.strip().split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def load_config(path: str, overrides=None) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides)


def allow_reid_checkpoint_globals():
    """Let ``torch.load(weights_only=True)`` (default since PyTorch 2.6) read the OSNet
    Re-ID checkpoint, which stores NumPy scalars and dtypes alongside its tensors."""
    import numpy as np
    import torch

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(
            [np.core.multiarray.scalar, np.dtype]
            + [type(np.dtype(t)) for t in ("float32", "float64", "int32", "int64")]
        )


_DEVICE = None


def get_device(cfg: dict = None):
    """Torch device from ``cfg['DEVICE']`` (``auto``, ``cuda``, ``cuda:1``, ``cpu``).

    Falls back to the CPU, with a warning, when CUDA is requested but unavailable.
    Without ``cfg``, returns the device of the configuration loaded by
    :func:`parse_step_args` (or ``auto``).
    """
    global _DEVICE
    import torch

    if cfg is None:
        if _DEVICE is not None:
            return _DEVICE
        cfg = {}
    requested = str(cfg.get("DEVICE", "auto") or "auto").lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: DEVICE={requested} but CUDA is not available, running on the CPU (very slow).",
              flush=True)
        requested = "cpu"
    _DEVICE = torch.device(requested)
    return _DEVICE


@dataclass
class SequencePaths:
    """All the files read or written for one sequence (one match half)."""

    name: str
    img_dir: str
    work_dir: str

    def _f(self, filename: str) -> str:
        return os.path.join(self.work_dir, filename)

    @property
    def homography_dir(self) -> str:
        return self._f("homographies")

    @property
    def tracks_raw(self) -> str:  # tracking + CLIP attributes
        return self._f("tracks_raw.txt")

    @property
    def tracks_verified(self) -> str:  # jersey number / color re-checked by Qwen
        return self._f("tracks_verified.txt")

    def tracks_attributes(self, cfg: dict) -> str:
        """Input of the refinement stage: Qwen-verified tracks unless Qwen is disabled."""
        return self.tracks_verified if cfg.get("QWEN", {}).get("ENABLED", True) else self.tracks_raw

    @property
    def tracks_dedup(self) -> str:  # duplicated boxes removed
        return self._f("tracks_dedup.txt")

    @property
    def tracklets(self) -> str:  # Re-ID features per tracklet (pickle)
        return self._f("tracklets.pkl")

    @property
    def tracks_refined(self) -> str:  # after ID-aware tracklet refinement
        return self._f("tracks_refined.txt")

    @property
    def tracks_pitch(self) -> str:  # refined tracks projected on the pitch (meters)
        return self._f("tracks_pitch.txt")

    @property
    def predictions(self) -> str:  # final SoccerNet-GSR JSON
        return self._f(f"{self.name}.json")

    @property
    def visualization_dir(self) -> str:
        return self._f("visualization")


def sequence_paths(cfg: dict, sequence: str) -> SequencePaths:
    data_dir = resolve(cfg["DATA_DIR"])
    out_dir = resolve(cfg["OUTPUT_DIR"])
    split = cfg["SPLIT"]
    paths = SequencePaths(
        name=sequence,
        img_dir=os.path.join(data_dir, split, sequence, "img1"),
        work_dir=os.path.join(out_dir, split, sequence),
    )
    os.makedirs(paths.work_dir, exist_ok=True)
    return paths


def list_sequences(cfg: dict) -> list[str]:
    split_dir = os.path.join(resolve(cfg["DATA_DIR"]), cfg["SPLIT"])
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    return sorted(
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d, "img1"))
    )


def parse_step_args(description: str, extra=None):
    """Parse the common step arguments; ``extra(parser)`` may add step-specific ones.

    Returns ``(args, cfg, paths)``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=os.path.join(BASELINE_DIR, "configs", "baseline.yaml"),
                        help="Baseline YAML configuration")
    parser.add_argument("--sequence", required=True,
                        help="Sequence folder name under DATA_DIR/SPLIT (one match half)")
    parser.add_argument("--set", dest="overrides", nargs="+", default=[], metavar="KEY=VALUE",
                        help="Override config entries, e.g. --set SPLIT=challenge QWEN.ENABLED=false")
    if extra is not None:
        extra(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    get_device(cfg)
    return args, cfg, sequence_paths(cfg, args.sequence)
