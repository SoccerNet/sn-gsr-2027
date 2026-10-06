# Data

The 2027 release adds **one challenge set**: the continuously annotated match half, including added time. Download its inputs from [SoccerNet/SN-GSR-2027](https://huggingface.co/datasets/SoccerNet/SN-GSR-2027):

```bash
hf download SoccerNet/SN-GSR-2027 --repo-type dataset --local-dir ./SN-GSR-2027
```

For development, use the unchanged [SoccerNet/SN-GSR-2025](https://huggingface.co/datasets/SoccerNet/SN-GSR-2025) `train.zip`, `valid.zip`, and `test.zip` files:

```bash
hf download SoccerNet/SN-GSR-2025 train.zip valid.zip test.zip --repo-type dataset --local-dir ./SN-GSR-2025
```

No new 2027 train, validation, or test split is introduced. Challenge annotations are withheld for evaluation. See the [evaluation guide](../evaluation/README.md) for the GS-HOTA input format.
