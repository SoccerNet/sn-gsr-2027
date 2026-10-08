# Data

The 2027 release adds **one challenge set**: the continuously annotated match half, including added time. Download its inputs from [SoccerNet/SN-GSR-2027](https://huggingface.co/datasets/SoccerNet/SN-GSR-2027):

```bash
hf download SoccerNet/SN-GSR-2027 --repo-type dataset --local-dir ./SN-GSR-2027
```

For development, use the unchanged [SoccerNet/SN-GSR-2025](https://huggingface.co/datasets/SoccerNet/SN-GSR-2025) `train.zip`, `valid.zip`, and `test.zip` files:

```bash
hf download SoccerNet/SN-GSR-2025 train.zip valid.zip test.zip --repo-type dataset --local-dir ./SN-GSR-2025
```

Extract the development splits into the SoccerNetGS layout used by the [TrackLab baseline](../baselines/README.md):

```bash
mkdir -p SoccerNetGS/train SoccerNetGS/valid SoccerNetGS/test
unzip -q SN-GSR-2025/train.zip -d SoccerNetGS/train
unzip -q SN-GSR-2025/valid.zip -d SoccerNetGS/valid
unzip -q SN-GSR-2025/test.zip -d SoccerNetGS/test
```

Each sequence has video frames and `Labels-GameState.json` annotations. The published GSR annotation version is 1.3; check `info.version` in the labels when preparing development data. No new 2027 train, validation, or test split is introduced. Treat the full 2027 match half as **one challenge sequence** so track identities remain continuous. Challenge annotations are withheld for evaluation. See the [evaluation guide](../evaluation/README.md) for the GS-HOTA input format.
