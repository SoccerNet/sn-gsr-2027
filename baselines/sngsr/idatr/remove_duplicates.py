"""IDATR step 1/4 - remove duplicated boxes (two overlapping tracks on the same person)."""
from itertools import combinations

import numpy as np
from loguru import logger
from tqdm import tqdm

def compute_iou(array1, array2):
    """
    Computes IoU between bounding boxes in array1 and array2.

    array1: (1, 10, 4) - bounding boxes for a single track
    array2: (12, 10, 4) - bounding boxes for multiple tracks
    threshold: IoU threshold for suppression

    Returns:
        iou_mask: (12, 10, 1) - Boolean mask indicating where IoU >= threshold
    """
    # Extract coordinates for broadcasting
    x1_1, y1_1, w1, h1 = array1
    x1_2, y1_2, w2, h2 = array2
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1  # Convert width & height to bottom-right coordinates
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    # Compute intersection coordinates
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    # Compute intersection area
    intersection_width = max(x_right - x_left, 0)
    intersection_height = max(y_bottom - y_top, 0)
    intersection_area = intersection_width * intersection_height  # (12, 10, 1)

    # Compute areas of bounding boxes
    area1 = w1 * h1  # (1, 10, 1)
    area2 = w2 * h2  # (12, 10, 1)

    # Compute IoU
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area if union_area > 0 else 0

def remove_double_bbox(cfg, paths, iou_thres=0.15, ver_dist_thres=3):
    """Drop, in every frame, the younger of two same-color tracks that overlap.

    Every frame of the sequence is processed (a whole match half, not a 30 s clip).
    """
    dtype = [
        ("FrameID", int),
        ("TrackID", int),
        ("X", float),
        ("Y", float),
        ("W", float),
        ("H", float),
        ("Score", float),
        ("Role", "U16"),
        ("JerseyNumber", int),
        ("JerseyColor", "U16"),
        ("Team", "U10"),
    ]
    src = paths.tracks_attributes(cfg)
    data = np.genfromtxt(src, delimiter=",", dtype=dtype, encoding="utf-8", ndmin=1)
    logger.info(f"Tracklets before removing double boxes: {len(set(data['TrackID']))}")

    # Group row indices by frame once (O(N log N)) instead of scanning the whole
    # file for every frame.
    order = np.argsort(data["FrameID"], kind="stable")
    frames, starts = np.unique(data["FrameID"][order], return_index=True)
    bounds = list(starts[1:]) + [len(order)]

    rmv_lines = set()
    for frame_id, start, end in tqdm(zip(frames, starts, bounds), total=len(frames), desc="Double boxes"):
        rows = order[start:end]
        if len(rows) < 2:
            continue
        frame = data[rows]
        boxes = np.column_stack((frame["X"], frame["Y"], frame["W"], frame["H"]))
        y_centers = frame["Y"] + frame["H"] / 2
        colors = np.char.lower(np.char.strip(frame["JerseyColor"]))
        for i, j in combinations(range(len(rows)), 2):
            if (colors[i] == colors[j] and abs(y_centers[i] - y_centers[j]) < ver_dist_thres
                    and compute_iou(boxes[i], boxes[j]) > iou_thres):
                rmv_lines.add(int(rows[j]))  # j was created after i: it is removed

    with open(src, "r") as f:
        lines = f.readlines()
    with open(paths.tracks_dedup, "w") as f:
        f.writelines(line for k, line in enumerate(lines) if k not in rmv_lines)

    kept = np.delete(data, sorted(rmv_lines))
    logger.info(f"Removed {len(rmv_lines)} double boxes; tracklets left: {len(set(kept['TrackID']))}")


if __name__ == "__main__":
    from sngsr.utils import parse_step_args

    _, cfg, paths = parse_step_args("IDATR 1/4: remove double bounding boxes.")
    remove_double_bbox(cfg, paths)
