# Long Form GSR baseline

The [SoccerNet Game State Reconstruction TrackLab pipeline](https://github.com/SoccerNet/sn-gamestate) is the reference starting point for athlete tracking and identification. It accompanies the [SoccerNet GSR paper](../README.md#citation). The upstream repository provides environment, weights, and data instructions; its previous-edition entry point is:

```bash
uv run tracklab -cn soccernet
```

Use the [SN-GSR-2025 train, validation, and test splits](https://huggingface.co/datasets/SoccerNet/SN-GSR-2025) to develop the method. Download the new [SN-GSR-2027 challenge set](https://huggingface.co/datasets/SoccerNet/SN-GSR-2027) with `hf download SoccerNet/SN-GSR-2027 --repo-type dataset --local-dir ./SN-GSR-2027`. Use the full match half as the challenge input sequence, preserving athlete identities throughout it. Export the prediction JSON expected by the [GS-HOTA evaluation guide](../evaluation/README.md). The upstream pipeline was developed for shorter clips; consult its configuration and sequence handling when applying it to a full half.
