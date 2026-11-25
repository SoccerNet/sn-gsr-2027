# Third-party notices

| Component | Location | Source | License |
| --- | --- | --- | --- |
| Broadcast2Pitch (SoccerNet GSR 2025 winner, KIST) | `sngsr/` (calibration, tracking, CLIP attributes, IDATR, export), `exp/`, `assets/pitch/`, all four baseline checkpoints (`yolox_soccernet.pth.tar`, `sports_model.pth.tar-60`, `SoccernetGSR_EfficientNet_Best.pth`, `CLIP_Jersey.pth`) | https://github.com/yinmayoo185/SoccernetGSR | no license file upstream; reused with the authors' permission |
| YOLOX | `yolox/` (models, data, exp, utils) | https://github.com/Megvii-BaseDetection/YOLOX | Apache-2.0, see `yolox/LICENSE` |
| Deep-EIoU | `yolox/EIoU_tracker/` | https://github.com/hsiangwei0903/Deep-EIoU | no license file upstream |
| ByteTrack (base of Deep-EIoU) | `yolox/EIoU_tracker/` (tracker, matching, Kalman filter) | https://github.com/FoundationVision/ByteTrack | MIT, see `yolox/EIoU_tracker/LICENSE-ByteTrack` |
| BoT-SORT (base of Deep-EIoU) | `yolox/EIoU_tracker/` (Re-ID association) | https://github.com/NirAharon/BoT-SORT | MIT, see `yolox/EIoU_tracker/LICENSE-BoT-SORT` |
| GTA-Link (tracklet refinement) | `sngsr/idatr/` (adapted) | https://github.com/sjc042/gta-link | MIT, see `sngsr/idatr/LICENSE-GTA-Link` |
| Torchreid / OSNet | installed from GitHub at a pinned commit (`requirements-git.txt`) | https://github.com/KaiyangZhou/deep-person-reid | MIT |
| CLIP | installed from GitHub at a pinned commit (`requirements-git.txt`) | https://github.com/openai/CLIP | MIT |
| Qwen3-VL | downloaded from the Hugging Face Hub (8B FP8 by default; 8B BF16 or 2B as fallbacks) | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8 | Apache-2.0 |
| sn-trackeval | evaluation dependency | https://github.com/SoccerNet/sn-trackeval | MIT |

The detection, calibration, CLIP attribute and IDATR components are adapted from
Broadcast2Pitch, the winning solution of the SoccerNet Game State Reconstruction
2025 challenge by the Human Data Intelligence Lab, Korea Institute of Science
and Technology (KIST), with the permission of its authors. The changes made for
the long-form setting are:

| Area | Broadcast2Pitch | This baseline |
| --- | --- | --- |
| Pipeline | separate scripts, paths edited by hand | one `sngsr` package, one YAML config, `run_baseline.py` with per-step logs, `status.json` and resuming (`--from-step`) |
| Calibration | homography per frame | single pass, optional `SFR.FRAME_STRIDE`, invalid frames reuse the last valid homography (back-filled at the start) |
| Jersey / role | CLIP or a fine-tuned LLaMA (LoRA), frame by frame | batched CLIP frame by frame (LLaMA removed), then a new track-level **Qwen3-VL verification** step (dichotomous frame sampling, agreement voting) |
| IDATR | 30-second clips | indexed tracklet generation and pitch projection so that a whole half fits in memory and time |
| Export | JSON per clip | majority voting per track, `left` / `right` team sides, `image_id` read from `Labels-GameState.json`, compact JSON and `submission.zip` |
| Distribution | weights on Google Drive | weights on the gated Hugging Face dataset, Docker image, frame-extraction and visualization tools |
| Portability | single setup | `DEVICE` honored by every step (`cuda:N`, CPU fallback), BF16 / FP16 fallback of Qwen3-VL on GPUs without FP8 / BF16, `--set` config overrides, pinned dependencies, non-root Docker image, `tools/check_env.py` and `tools/make_subset.py` |
