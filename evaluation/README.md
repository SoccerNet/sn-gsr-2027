# Long Form GSR evaluation

The ranking metric is **GS-HOTA**, using the SoccerNet Game State Reconstruction evaluator from the `SoccerNet` Python package. It is backed by the [SoccerNet TrackEval fork](https://github.com/SoccerNet/sn-trackeval). The [scoring adapter](scoring.py) calls `SoccerNet.Evaluation.GameStateReconstruction.evaluate` and writes `GS-HOTA`, `GS-DetA`, `GS-AssA`, and `score` (equal to GS-HOTA) as fractions. It runs the package in an isolated temporary directory because the upstream evaluator uses a relative `./temp` path.

## ZIP inputs

The adapter reads `input/ref/reference.zip` and `input/res/predictions.zip`. The reference ZIP follows the SoccerNetGS layout with `<sequence>/Labels-GameState.json`; reference annotations stay in the scoring environment. Submit a prediction ZIP with one JSON file named for the full-half sequence:

```text
predictions.zip
└── <sequence>.json
```

The JSON has a top-level `predictions` list. Each detection uses the SoccerNetGS fields `image_id` (frame), `track_id`, `category_id`, `supercategory`, `attributes` (`role`, `team`, `jersey`), and `bbox_pitch`. A `confidence` field is optional; the evaluator uses 1 when it is omitted. The SoccerNet evaluator reads all six pitch-coordinate keys in `bbox_pitch`: `x_bottom_left`, `y_bottom_left`, `x_bottom_middle`, `y_bottom_middle`, `x_bottom_right`, and `y_bottom_right`, measured in meters. Set `supercategory` to `object` and `category_id` to `1.0` for athlete detections, following the [published GSR submission format](https://github.com/SoccerNet/sn-gamestate/blob/main/ChallengeRules.md#submission-format) and [evaluator parser](https://github.com/SoccerNet/sn-trackeval/blob/main/trackeval/datasets/soccernet_gs.py). Use the same frame IDs as the challenge input and keep track IDs stable across the full half. The [upstream sample prediction ZIP](https://github.com/SoccerNet/sn-gamestate/blob/main/examples_predictions/SoccerNetGS-test.zip) shows the format on an earlier split.

* `role` is one of `player`, `goalkeeper`, `referee`, `other` (or `ball`, ignored).
* `team` is `left` or `right` (with respect to the camera viewpoint) for players and goalkeepers, `null` for referees and others.
* `jersey` is the jersey number (string or integer) for players and goalkeepers, `null` when it is not visible in the video and for referees and others.

Install the evaluator and TrackEval fork:

```bash
python -m pip install SoccerNet
python -m pip install git+https://github.com/SoccerNet/sn-trackeval.git@9c25232f6f2b56c9f203f1eb55784ff1e97df683
```

Run from this repository root:

```bash
python evaluation/scoring.py --input /path/to/input --output /path/to/output --split challenge
```

The result is `/path/to/output/scores.json`; `score` equals `GS-HOTA`. Scores are fractions in `[0, 1]`. The full-half prediction should preserve identities across the sequence; GS-HOTA measures detection, localization, and association under the SoccerNet protocol.
