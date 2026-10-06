"""Call the SoccerNet pip evaluator from the adapter's isolated working dir."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from SoccerNet.Evaluation.GameStateReconstruction import evaluate


def main() -> None:
    reference, predictions, split, output = sys.argv[1:]
    metrics = evaluate(reference, predictions, split=split)
    metrics["score"] = metrics["GS-HOTA"]
    Path(output).write_text(
        json.dumps(metrics, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
