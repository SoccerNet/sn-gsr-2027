"""
Step 3 - Jersey number and color verification with Qwen3-VL.

CLIP reads jersey attributes frame by frame, which is noisy over a whole match
half. This step re-reads every player / goalkeeper track with Qwen3-VL:

  * frames of the track are sampled in dichotomous order (middle, quarters,
    eighths, ...), so that the samples spread over the whole track;
  * each crop is answered with (number, color); ambiguous views return the
    sentinel number 100 and are ignored (they never count as a vote);
  * as soon as one (number, color) pair collects ``MIN_AGREEMENT`` votes the
    track is validated, otherwise the most voted pair after ``MAX_ATTEMPTS``
    crops is used (fallback);
  * every row of the track is rewritten with the retained (number, color), and
    the team of players whose color changed follows the color -> team mapping
    decided by the tracking step.

Input: ``tracks_raw.txt``.  Output: ``tracks_verified.txt``.

Usage:
    python -m sngsr.jersey.verify --config configs/baseline.yaml --sequence <name>
"""

import os
from collections import Counter

import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm

from sngsr.jersey.qwen import NOT_VISIBLE, QwenJerseyReader
from sngsr.utils import TRACK_COLUMNS, get_device, parse_step_args

PLAYER_ROLES = {"player", "goalkeeper"}


def dichotomic_indices(n: int, max_count: int) -> list[int]:
    """Indices 0..n-1 visited in dichotomous order (middle, quarters, ...)."""
    if n <= 0:
        return []
    if n <= max_count:
        mid = n // 2
        order = [mid]
        for k in range(1, n):
            if mid + k < n:
                order.append(mid + k)
            if mid - k >= 0:
                order.append(mid - k)
        return order[:n]

    seen: set[int] = set()
    out: list[int] = []
    denom = 2
    while len(out) < max_count and denom <= 4 * n:
        for num in range(1, denom, 2):
            idx = (num * n) // denom
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
                if len(out) >= max_count:
                    return out
        denom *= 2
    return out


def crop_at(row, img_dir: str):
    """Load the frame of a track row and return the box crop as PIL, or None."""
    frame = cv2.imread(os.path.join(img_dir, f"{int(row['frame']):06d}.jpg"))
    if frame is None:
        return None
    x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))


def verify_track(reader, rows, img_dir, max_attempts, min_agreement, batch_size):
    """Return (number, color, attempts_used), or None if no vote was collected.

    Early stopping is checked between mini-batches, so up to batch_size-1 extra
    crops may be read on the validating batch.
    """
    rows = rows.reset_index(drop=True)
    order = dichotomic_indices(len(rows), max_attempts)
    votes: Counter = Counter()
    attempts = 0

    for start in range(0, len(order), batch_size):
        crops = [c for c in (crop_at(rows.iloc[i], img_dir) for i in order[start:start + batch_size])
                 if c is not None]
        if not crops:
            continue
        validated = None
        for number, color in reader.read(crops):
            attempts += 1
            if number == NOT_VISIBLE:
                continue
            votes[(number, color)] += 1
            if votes[(number, color)] >= min_agreement and validated is None:
                validated = (number, color)
        if validated is not None:
            return validated[0], validated[1], attempts

    if votes:
        (number, color), _ = votes.most_common(1)[0]
        return number, color, attempts
    return None


def build_color_to_team(df: pd.DataFrame) -> dict[str, int]:
    """Color -> team (0/1) as decided by the tracking step (dominant color of each team)."""
    players = df[(df["role_lc"] == "player") & df["team"].isin([0, 1])]
    mapping: dict[str, int] = {}
    for team in (0, 1):
        colors = players.loc[players["team"] == team, "color"]
        colors = colors[colors != "unknown"]
        if not colors.empty:
            c = colors.value_counts().idxmax()
            if c in mapping:  # same dominant color for both teams: ambiguous
                return {}
            mapping[c] = team
    return mapping


def run(cfg, paths):
    qcfg = cfg.get("QWEN", {})
    max_attempts = int(qcfg.get("MAX_ATTEMPTS", 96))
    min_agreement = int(qcfg.get("MIN_AGREEMENT", 3))
    batch_size = int(qcfg.get("BATCH_SIZE", 4))

    df = pd.read_csv(paths.tracks_raw, header=None, names=TRACK_COLUMNS)
    df["role_lc"] = df["role"].astype(str).str.lower()
    df["color"] = df["color"].astype(str).str.strip().str.lower()
    color2team = build_color_to_team(df)
    print(f"[qwen] color -> team map: {color2team}", flush=True)

    player_tids = df.loc[df["role_lc"].isin(PLAYER_ROLES), "track_id"].unique().tolist()
    print(f"[qwen] {len(player_tids)} player tracks to verify "
          f"(budget={max_attempts}, agreement={min_agreement})", flush=True)

    reader = QwenJerseyReader(qcfg.get("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct-FP8"), device=get_device(cfg))
    stats = Counter()
    for tid in tqdm(player_tids, desc="Qwen verification"):
        track_rows = df[(df["track_id"] == tid) & df["role_lc"].isin(PLAYER_ROLES)]
        orig_jn = track_rows["jersey"].mode().iloc[0]
        orig_color = track_rows["color"].mode().iloc[0]

        if len(track_rows) < min_agreement:  # cannot reach the agreement threshold
            stats["skipped_short"] += 1
            continue
        result = verify_track(reader, track_rows, paths.img_dir, max_attempts, min_agreement, batch_size)
        if result is None:
            stats["unchanged"] += 1
            continue

        number, color, attempts = result
        color = str(color).strip().lower()
        if color == "unknown":  # Qwen could not tell: keep the CLIP color
            color = orig_color
        mask = df["track_id"] == tid
        df.loc[mask, "jersey"] = number
        df.loc[mask, "color"] = color
        if color != orig_color and color in color2team:
            df.loc[mask & (df["role_lc"] == "player"), "team"] = color2team[color]

        stats["validated" if attempts < max_attempts else "fallback"] += 1
        stats["changed" if (str(orig_jn), orig_color) != (str(number), color) else "confirmed"] += 1

    df = df.drop(columns=["role_lc"])
    df["frame"] = df["frame"].astype(str).str.zfill(6)
    df.to_csv(paths.tracks_verified, index=False, header=False)
    print(f"[qwen] {dict(stats)}", flush=True)
    print(f"[qwen] Saved {paths.tracks_verified}", flush=True)


if __name__ == "__main__":
    _, cfg, paths = parse_step_args("Jersey number / color verification with Qwen3-VL.")
    run(cfg, paths)
