# Evaluation — GS-HOTA on a continuous half

The 2027 long-form task uses the **same GS-HOTA metric and SoccerNet TrackEval implementation** as earlier Game State Reconstruction editions. The difference is the evaluation unit: one continuously annotated match half (45 minutes plus added time), rather than the 200 separate 30-second clips drawn from 11 games. This is a much longer association problem; tracker IDs must remain coherent across the complete half according to the final task rules.

## Exact evaluator

The preferred [scoring.py](scoring.py) adapter calls
`SoccerNet.Evaluation.GameStateReconstruction.evaluate` from the SoccerNet
Python package. This is the previous-edition GS-HOTA entry point and returns
`GS-HOTA`, `GS-DetA`, and `GS-AssA` as fractions. `score` is `GS-HOTA`.
The package creates and deletes `./temp`, so the adapter runs it from a
private temporary directory.

Install the SoccerNet package and its [TrackEval fork](https://github.com/SoccerNet/sn-trackeval)
in the scoring image:

```bash
python -m pip install SoccerNet
python -m pip install git+https://github.com/SoccerNet/sn-trackeval.git@9c25232f6f2b56c9f203f1eb55784ff1e97df683
```

Then prepare `ref/reference.zip` (organizer-only
ground truth) and `res/predictions.zip` under an input directory. From this
repository root, run:

```bash
python evaluation/scoring.py --input /path/to/input --output /path/to/output --split challenge
```

The result is `/path/to/output/scores.json`. The upstream package requires
GT JSON in its legacy ZIP directory layout and prediction JSON files named
by sequence. The exact one-half sequence name and ZIP content have not yet
been tested. The [run_gs_hota.py](run_gs_hota.py) launcher below remains
available for direct TrackEval debugging.

The [SoccerNet TrackEval fork](https://github.com/SoccerNet/sn-trackeval) implements GS-HOTA in its `SoccerNetGS` dataset adapter and HOTA metric. In its output the official GS-HOTA value appears as **`HOTA`**. Pin and verify the fork at commit `9c25232f6f2b56c9f203f1eb55784ff1e97df683` before producing reference scores. [run_gs_hota.py](run_gs_hota.py) is a thin local launcher; it does not reimplement the metric.

The previous input layout is:

```text
<GT_ROOT>/<split>/<sequence>/Labels-GameState.json
<PRED_ROOT>/SoccerNetGS-<split>/<tracker>/<sequence>.json
```

After installing the fork and preparing data in that layout, run:

```bash
python3 evaluation/run_gs_hota.py \
  --trackeval /path/to/sn-trackeval \
  --gt-folder /path/to/ground_truth \
  --predictions-folder /path/to/predictions \
  --split valid \
  --tracker my_method
```

This uses the upstream scorer's default pitch-space Gaussian location similarity and role, team, and jersey-number matching, with `HOTA` as the output metric. The final 2027 sequence name, JSON schema, frame IDs, label flags, and GT/prediction conversion must be validated against the fully annotated half. Do not compare unverified long-form scores with the short-clip benchmark.

## Remaining work for CodaBench

Freeze the 2027 split manifests and one-half annotation format, verify the upstream evaluator processes the full sequence without frame drops or memory issues, assemble a private reference package, and score a known baseline. Challenge ground truth remains outside Git. The [data layout](../data/README.md) tracks the public split status.
