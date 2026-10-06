# Long Form GSR evaluation

The ranking metric is **GS-HOTA**, using the SoccerNet Game State Reconstruction evaluator from the `SoccerNet` Python package. It is backed by the [SoccerNet TrackEval fork](https://github.com/SoccerNet/sn-trackeval). The [scoring adapter](scoring.py) calls `SoccerNet.Evaluation.GameStateReconstruction.evaluate` and writes `GS-HOTA`, `GS-DetA`, `GS-AssA`, and `score` (equal to GS-HOTA) as fractions. It runs the package in an isolated temporary directory because the upstream evaluator uses a relative `./temp` path.

## ZIP inputs

The adapter reads `input/ref/reference.zip` and `input/res/predictions.zip`. The reference ZIP follows the SoccerNetGS layout with `<sequence>/Labels-GameState.json`; prediction JSON files are named `<sequence>.json`. The sequence covers the match half. Reference annotations stay in the scoring environment.

Install the evaluator and TrackEval fork:

```bash
python -m pip install SoccerNet
python -m pip install git+https://github.com/SoccerNet/sn-trackeval.git@9c25232f6f2b56c9f203f1eb55784ff1e97df683
```

Run from this repository root:

```bash
python evaluation/scoring.py --input /path/to/input --output /path/to/output --split challenge
```

The result is `/path/to/output/scores.json`. For direct TrackEval use on unpacked data, [run_gs_hota.py](run_gs_hota.py) accepts the TrackEval checkout, ground-truth directory, prediction directory, split, and tracker name. Its directory layout is:

```text
<GT_ROOT>/<split>/<sequence>/Labels-GameState.json
<PRED_ROOT>/SoccerNetGS-<split>/<tracker>/<sequence>.json
```

The full-half prediction should preserve identities across the sequence; GS-HOTA measures detection, localization, and association under the SoccerNet protocol.
