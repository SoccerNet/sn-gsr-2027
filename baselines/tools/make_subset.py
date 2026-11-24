"""Cut a short excerpt of a match half to check an installation in minutes.

A full half takes many hours; this tool links N consecutive frames of a half
into a separate split (``<SPLIT>-subset`` by default) and writes a matching
``Labels-GameState.json`` (frames renumbered from 000001, original image ids
kept), so the whole pipeline and the scoring can be run end to end on it:

    python tools/make_subset.py --set DATA_DIR=/absolute/path/to/SoccerNetGS SPLIT=valid \\
        --sequence SNGS-XXXX --num-frames 750
    python run_baseline.py --set DATA_DIR=/absolute/path/to/SoccerNetGS SPLIT=valid-subset \\
        --sequences SNGS-XXXX

then score ``outputs/valid-subset/submission.zip`` as described in
``evaluation/README.md``.

Frames are relative symbolic links (no copy), so the excerpt also works inside
Docker as long as the whole DATA_DIR is mounted. Scores on an excerpt are only
a sanity check; they are not comparable to full-half scores.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sngsr.utils import BASELINE_DIR, load_config, resolve  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(BASELINE_DIR, "configs", "baseline.yaml"))
    parser.add_argument("--set", dest="overrides", nargs="+", default=[], metavar="KEY=VALUE")
    parser.add_argument("--sequence", required=True, help="Source half under DATA_DIR/SPLIT")
    parser.add_argument("--start-frame", type=int, default=1, help="First source frame (1-based)")
    parser.add_argument("--num-frames", type=int, default=750, help="Number of frames (750 = 30 s)")
    parser.add_argument("--dst-split", default=None, help="Destination split (default: <SPLIT>-subset)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    data_dir, split = resolve(cfg["DATA_DIR"]), cfg["SPLIT"]
    src = os.path.join(data_dir, split, args.sequence)
    dst = os.path.join(data_dir, args.dst_split or f"{split}-subset", args.sequence)
    src_img, dst_img = os.path.join(src, "img1"), os.path.join(dst, "img1")
    if not os.path.isdir(src_img):
        sys.exit(f"Frames not found: {src_img}")
    if os.path.exists(dst):
        sys.exit(f"{dst} already exists; remove it first")

    first, last = args.start_frame, args.start_frame + args.num_frames - 1
    os.makedirs(dst_img)
    linked = 0
    for new, old in enumerate(range(first, last + 1), start=1):
        src_frame = os.path.join(src_img, f"{old:06d}.jpg")
        if not os.path.isfile(src_frame):
            break
        os.symlink(os.path.relpath(src_frame, dst_img), os.path.join(dst_img, f"{new:06d}.jpg"))
        linked += 1
    if linked == 0:
        sys.exit(f"No frame in [{first}, {last}] in {src_img}")

    labels_path = os.path.join(src, "Labels-GameState.json")
    if os.path.isfile(labels_path):
        with open(labels_path) as f:
            labels = json.load(f)
        images, kept = [], set()
        for im in labels.get("images", []):
            frame = int(os.path.splitext(im["file_name"])[0])
            if first <= frame < first + linked:
                im = dict(im, file_name=f"{frame - first + 1:06d}.jpg")
                images.append(im)
                kept.add(im["image_id"])
        labels["images"] = images
        if "annotations" in labels:
            labels["annotations"] = [a for a in labels["annotations"] if a["image_id"] in kept]
        if isinstance(labels.get("info"), dict) and "seq_length" in labels["info"]:
            labels["info"]["seq_length"] = linked
        with open(os.path.join(dst, "Labels-GameState.json"), "w") as f:
            json.dump(labels, f)

    print(f"{linked} frames ({first}..{first + linked - 1}) of {args.sequence} -> {dst}")
    print(f"Run it with: python run_baseline.py --set SPLIT={os.path.basename(os.path.dirname(dst))} "
          f"--sequences {args.sequence}")


if __name__ == "__main__":
    main()
