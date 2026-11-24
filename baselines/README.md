# Long Form GSR baseline

The reference starting point is a modular, resumable pipeline that reconstructs the game state of a **whole match half** and writes a ready-to-submit ZIP. It is derived from [Broadcast2Pitch](https://github.com/yinmayoo185/SoccernetGSR), the winning solution of the SoccerNet Game State Reconstruction 2025 challenge (Human Data Intelligence Lab, KIST), whose code and weights are reused with the permission of its authors. It chains camera calibration (one homography per frame), YOLOX detection with Deep-EIoU tracking and OSNet Re-ID, CLIP role, jersey-number and color heads, Qwen3-VL jersey verification per track, ID-aware tracklet refinement (IDATR), pitch projection and SoccerNet JSON export. Every step is a separate, resumable stage, so any of them can be replaced by your own method. Third-party components and licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Requirements: Linux (or WSL2), one NVIDIA GPU, Python ≥ 3.10, `ffmpeg` (only for `--video`), and about 50 GB of disk per half for the frames plus about 10 GB per half for the outputs. The released configuration runs Qwen3-VL-8B in FP8 (GPU compute capability ≥ 8.9, ≥ 16 GB); on Ampere GPUs with ≥ 24 GB the BF16 version is used automatically. On smaller GPUs, use `--set QWEN.MODEL_ID=Qwen/Qwen3-VL-2B-Instruct` or `--set QWEN.ENABLED=false`; these settings change the jersey verification and therefore the scores.

Download and extract the [SN-GSR-2025 development splits](../data/README.md). Then build the Docker image, which pins the CUDA and PyTorch stack and requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and an NVIDIA driver ≥ 580:

```bash
cd baselines
docker build -t sngsr-baseline .
mkdir -p checkpoints outputs ~/.cache/sngsr
alias sngsr='docker run --rm -it --gpus all --ipc=host --user "$(id -u):$(id -g)" \
    -v /absolute/path/to/data:/workspace/data \
    -v "$PWD/checkpoints:/workspace/baselines/checkpoints" \
    -v "$PWD/outputs:/workspace/baselines/outputs" \
    -v "$HOME/.cache/sngsr:/cache" \
    sngsr-baseline'
```

Prefix the commands below with `sngsr`; inside the container, `DATA_DIR` paths start with `/workspace/data/`. Without Docker, install PyTorch for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/), then:

```bash
cd baselines
pip install numpy cython
pip install -r requirements.txt --no-build-isolation
pip install -r requirements-git.txt --no-build-isolation --no-deps
pip install -e .
```

`python tools/check_env.py` checks the packages, the GPU, the checkpoints, the Hugging Face login, the data and the free disk space. The four baseline checkpoints are redistributed on the gated SN-GSR-2027 dataset; the CLIP ViT-L/14 backbone and [Qwen3-VL-8B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8) are downloaded automatically on first use:

```bash
hf auth login
python tools/download_weights.py
```

The pipeline reads frames from `DATA_DIR/SPLIT/<sequence>/img1/` and writes everything to `outputs/SPLIT/<sequence>/`. Point `DATA_DIR` to the absolute path of the directory containing the split folders. To run on the validation split of the `SoccerNetGS` development data:

```bash
python run_baseline.py --set DATA_DIR=/absolute/path/to/SoccerNetGS SPLIT=valid
```

Every entry of [configs/baseline.yaml](configs/baseline.yaml) can be overridden with `--set KEY=VALUE` (dots for nested keys), and `--sequences` restricts the run to given sequences. Finished steps are recorded in `status.json` and skipped when the same command is launched again; `--from-step <step>` recomputes from a given step. The run writes one `<sequence>.json` per sequence and gathers them in `outputs/SPLIT/submission.zip`, the format accepted by the [evaluation guide](../evaluation/README.md).

Download the [SN-GSR-2027 challenge input](https://huggingface.co/datasets/SoccerNet/SN-GSR-2027) from Hugging Face and process the full match half as one sequence, which is the default `SPLIT`:

```bash
python run_baseline.py --set DATA_DIR=/absolute/path/to/SN-GSR-2027 SPLIT=challenge
```

The export reads the `image_id` of every frame from the half's `Labels-GameState.json` and keeps track identities across the whole half. A full half takes many hours (about 32 h on an NVIDIA GB10 before the speed-ups of this release); `SFR.FRAME_STRIDE` > 1 and a smaller `QWEN.MAX_ATTEMPTS` reduce it, and `--set DEVICE=cuda:N` selects the GPU.
