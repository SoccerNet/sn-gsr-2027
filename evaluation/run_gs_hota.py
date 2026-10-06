"""Run the unchanged SoccerNet GS-HOTA implementation on a 2027 split.

This wrapper supplies the directory layout used by the upstream SoccerNet TrackEval
fork. It does not redefine or reimplement GS-HOTA. The 2027 annotation adapter
must be validated before official scoring.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

def build_command(args):
    runner = args.trackeval / "scripts" / "run_soccernet_gs.py"
    if not runner.is_file():
        raise FileNotFoundError(f"SoccerNet TrackEval runner not found: {runner}")
    if not args.gt_folder.is_dir() or not args.predictions_folder.is_dir():
        raise FileNotFoundError("Ground-truth or prediction directory not found")
    return [
        sys.executable, str(runner),
        "--GT_FOLDER", str(args.gt_folder),
        "--TRACKERS_FOLDER", str(args.predictions_folder),
        "--TRACKER_SUB_FOLDER", "",
        "--SPLIT_TO_EVAL", args.split,
        "--TRACKERS_TO_EVAL", args.tracker,
        "--METRICS", "HOTA",
        "--USE_PARALLEL", "False",
        "--PRINT_CONFIG", "False",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trackeval", type=Path, required=True, help="Checkout of SoccerNet/sn-trackeval")
    parser.add_argument("--gt-folder", type=Path, required=True, help="Root containing <split>/<sequence>/Labels-GameState.json")
    parser.add_argument("--predictions-folder", type=Path, required=True, help="Root containing SoccerNetGS-<split>/<tracker>/<sequence>.json")
    parser.add_argument("--split", required=True, choices=("train", "valid", "test", "challenge"))
    parser.add_argument("--tracker", required=True, help="Name of the prediction subdirectory")
    args = parser.parse_args()
    return subprocess.call(build_command(args))


if __name__ == "__main__":
    raise SystemExit(main())
