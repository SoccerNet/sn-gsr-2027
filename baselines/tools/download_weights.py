"""Download the baseline checkpoints from the SN-GSR-2027 Hugging Face dataset.

The dataset is gated: request access on
https://huggingface.co/datasets/SoccerNet/SN-GSR-2027 and run ``hf auth login`` first.

    python tools/download_weights.py                      # -> baselines/checkpoints/

The CLIP ViT-L/14 backbone (OpenAI) and Qwen3-VL are downloaded automatically
on first use, to ~/.cache/clip and ~/.cache/huggingface respectively.
"""

import argparse
import os
import shutil

from huggingface_hub import hf_hub_download

BASELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CHECKPOINTS = {
    "yolox_soccernet.pth.tar": "YOLOX-X person/ball detector",
    "sports_model.pth.tar-60": "OSNet Re-ID (tracking and tracklet refinement)",
    "SoccernetGSR_EfficientNet_Best.pth": "pitch keypoints / lines (camera calibration)",
    "CLIP_Jersey.pth": "CLIP role / jersey number / jersey color heads",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default="SoccerNet/SN-GSR-2027")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--remote-dir", default="baseline/checkpoints",
                        help="Folder of the checkpoints inside the Hugging Face repository")
    parser.add_argument("--output-dir", default=os.path.join(BASELINE_DIR, "checkpoints"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for name, description in CHECKPOINTS.items():
        target = os.path.join(args.output_dir, name)
        if os.path.isfile(target):
            print(f"[skip] {name} already present")
            continue
        print(f"[get ] {name} ({description})")
        path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            filename=f"{args.remote_dir}/{name}",
        )
        shutil.copyfile(path, target)
    print(f"Checkpoints ready in {args.output_dir}")


if __name__ == "__main__":
    main()
