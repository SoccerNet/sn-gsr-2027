# Data

Download the match-half data from [SoccerNet/SN-GSR-2027](https://huggingface.co/datasets/SoccerNet/SN-GSR-2027):

```bash
hf download SoccerNet/SN-GSR-2027 --repo-type dataset --local-dir ./SN-GSR-2027
```

The task covers one continuously annotated match half, including added time. Challenge annotations are withheld for evaluation. See the [evaluation guide](../evaluation/README.md) for the GS-HOTA input format.
