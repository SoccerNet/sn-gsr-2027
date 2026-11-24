"""
Optional - render the predictions of a match half as an annotated MP4.

Boxes are drawn on the broadcast frames (red: left team, blue: right team,
yellow: referees / others) and the pitch positions on a minimap in the top
right corner. Frames are streamed to ffmpeg, so no intermediate image is
written to disk.

Usage:
    python -m sngsr.visualize --config configs/baseline.yaml --sequence <name> \
        [--start-frame 1] [--num-frames 1500]
"""

import json
import os
import subprocess
from collections import defaultdict

import cv2
from tqdm import tqdm

from sngsr.utils import parse_step_args, resolve

LEFT, RIGHT, OTHER = (0, 0, 255), (255, 0, 0), (0, 255, 255)  # BGR


def draw_frame(image, predictions, radar, center):
    radar = radar.copy()
    for p in predictions:
        attributes = p.get("attributes", {})
        role = str(attributes.get("role", "")).lower()
        if role in ("referee", "other", "ball"):
            color, label = OTHER, role.capitalize()
        else:
            team = str(attributes.get("team", "")).lower()
            color = LEFT if team == "left" else RIGHT if team == "right" else OTHER
            jersey = attributes.get("jersey")
            label = f"ID:{p['track_id']} J:{jersey if jersey is not None else '-'} {team[:1].upper()}"

        b = p.get("bbox_image")
        if b:
            x, y, w, h = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        b = p.get("bbox_pitch")
        if b:
            cv2.circle(radar, (int(b["x_bottom_middle"] + center[0]), int(b["y_bottom_middle"] + center[1])),
                       1, color, -1)

    radar = cv2.resize(radar, (image.shape[1] // 4, image.shape[0] // 4))
    x0 = image.shape[1] - radar.shape[1] - 10
    image[10:10 + radar.shape[0], x0:x0 + radar.shape[1]] = radar
    return image


def render(cfg, paths, start_frame=1, num_frames=None, fps=None, crf=23):
    with open(paths.predictions) as f:
        by_frame = defaultdict(list)
        for p in json.load(f)["predictions"]:
            by_frame[int(p["frame"])].append(p)

    radar = cv2.imread(resolve(cfg["SFR"]["TEMPLATE_PATH"]))
    center = (radar.shape[1] / 2, radar.shape[0] / 2)
    frames = sorted(int(f[:-4]) for f in os.listdir(paths.img_dir) if f.endswith(".jpg"))
    frames = [f for f in frames if f >= start_frame]
    if num_frames:
        frames = frames[:num_frames]
    if not frames:
        raise ValueError("No frame to render")

    first = cv2.imread(os.path.join(paths.img_dir, f"{frames[0]:06d}.jpg"))
    h, w = first.shape[:2]
    out = os.path.join(paths.work_dir, f"{paths.name}_annotated.mp4")
    ffmpeg = subprocess.Popen([
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
        "-r", str(fps or cfg["TRACKER"]["FPS"]), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", out,
    ], stdin=subprocess.PIPE)
    for frame in tqdm(frames, desc=f"Rendering {paths.name}"):
        image = cv2.imread(os.path.join(paths.img_dir, f"{frame:06d}.jpg"))
        if image is None:
            continue
        ffmpeg.stdin.write(draw_frame(image, by_frame.get(frame, []), radar, center).tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"Saved {out}")


if __name__ == "__main__":
    def extra(p):
        p.add_argument("--start-frame", type=int, default=1)
        p.add_argument("--num-frames", type=int, default=None, help="Default: until the end of the half")
        p.add_argument("--crf", type=int, default=23)

    args, cfg, paths = parse_step_args("Render the predictions as an annotated MP4.", extra)
    render(cfg, paths, args.start_frame, args.num_frames, crf=args.crf)
