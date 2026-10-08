# Long Form GSR baseline

The reference starting point is the [SoccerNet Game State Reconstruction TrackLab pipeline](https://github.com/SoccerNet/sn-gamestate), which accompanies the [GSR paper](../README.md#citation). It combines athlete detection, re-identification and tracking with pitch calibration, role, team, and jersey-number modules. The implementation and pretrained component weights are maintained upstream.

Download and extract the [SN-GSR-2025 development splits](../data/README.md). Then install the upstream pipeline using its documented `uv` setup:

```bash
git clone https://github.com/SoccerNet/sn-gamestate.git
cd sn-gamestate
uv venv --python 3.9
uv pip install -e .
uv run mim install mmcv==2.0.1
```

Point `data_dir` to the absolute path of the directory containing `SoccerNetGS`. The default run processes one validation sequence and downloads pretrained weights as needed:

```bash
uv run tracklab -cn soccernet data_dir=/absolute/path/to/data
```

To reproduce the upstream baseline across the complete validation split, use the configuration's `dataset.nvid=-1` and `dataset.eval_set=valid` controls:

```bash
uv run tracklab -cn soccernet data_dir=/absolute/path/to/data dataset.eval_set=valid dataset.nvid=-1
```

The upstream pipeline writes a SoccerNetGS prediction ZIP under its run's `eval/pred/` directory. This is the format accepted by the [evaluation guide](../evaluation/README.md). The [SoccerNet configuration](https://github.com/SoccerNet/sn-gamestate/blob/main/sn_gamestate/configs/soccernet.yaml) describes the dataset and module settings.

Download the [SN-GSR-2027 challenge input](https://huggingface.co/datasets/SoccerNet/SN-GSR-2027) from Hugging Face. Process the full match half as one sequence and retain track identities across it. Export one JSON file with the same sequence name and the prediction fields described in the [evaluation guide](../evaluation/README.md). The longer input may require adjusting the pipeline's video selection, batch sizes, and retained tracker state; the [upstream configuration](https://github.com/SoccerNet/sn-gamestate/blob/main/sn_gamestate/configs/soccernet.yaml) documents these controls.
