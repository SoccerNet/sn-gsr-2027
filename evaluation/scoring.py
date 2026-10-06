"""CodaBench adapter for SoccerNet's published GameStateReconstruction evaluator.

The upstream package uses and deletes ./temp, so run it from an isolated
temporary working directory. The full-half 2027 package format is pending.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/app/input"))
    parser.add_argument("--output", type=Path, default=Path("/app/output"))
    parser.add_argument("--split", default="challenge")
    args = parser.parse_args()
    reference = (args.input / "ref" / "reference.zip").resolve()
    predictions = (args.input / "res" / "predictions.zip").resolve()
    if not reference.is_file() or not predictions.is_file():
        parser.error("Expected ref/reference.zip and res/predictions.zip")
    args.output.mkdir(parents=True, exist_ok=True)
    scores = (args.output / "scores.json").resolve()
    scores.unlink(missing_ok=True)
    worker = Path(__file__).with_name("soccernet_worker.py").resolve()
    with TemporaryDirectory(prefix="sn-gsr-eval-") as workdir:
        result = subprocess.run(
            [sys.executable, str(worker), str(reference), str(predictions),
             args.split, str(scores)],
            cwd=workdir, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
    print(result.stdout[-12000:])
    if result.returncode:
        scores.unlink(missing_ok=True)
        return result.returncode
    try:
        payload = json.loads(scores.read_text(encoding="utf-8"))
        assert all(isinstance(payload[key], (int, float))
                   for key in ("score", "GS-HOTA", "GS-DetA", "GS-AssA"))
    except (OSError, ValueError, KeyError, AssertionError):
        scores.unlink(missing_ok=True)
        print("SoccerNet evaluator did not produce valid scores.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
