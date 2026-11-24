"""Extract the frames of a match-half video into the layout expected by the baseline.

    python tools/extract_frames.py --video half1.mp4 --output ../data/SN-GSR-2027/challenge/SNGS-0001

writes ``<output>/img1/000001.jpg, 000002.jpg, ...`` at 25 fps (frame numbers
start at 1, as in SoccerNet-GSR). A match half is roughly 40-50 GB of JPEGs at
1080p with the default quality.
"""

import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True, help="Sequence folder; frames go to <output>/img1/")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--quality", type=int, default=2, help="ffmpeg -q:v (2 = best, 31 = worst)")
    args = parser.parse_args()

    img_dir = os.path.join(args.output, "img1")
    os.makedirs(img_dir, exist_ok=True)
    if any(f.endswith(".jpg") for f in os.listdir(img_dir)):
        raise SystemExit(f"{img_dir} already contains frames")
    subprocess.run([
        "ffmpeg", "-v", "error", "-stats", "-i", args.video, "-an",
        "-vf", f"fps={args.fps}", "-q:v", str(args.quality), "-start_number", "1",
        os.path.join(img_dir, "%06d.jpg"),
    ], check=True)
    print(f"{len(os.listdir(img_dir))} frames written to {img_dir}")


if __name__ == "__main__":
    main()
