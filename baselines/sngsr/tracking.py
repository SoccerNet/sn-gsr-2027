"""
Step 2 - Detection, tracking and attribute recognition.

For every frame of a match half:
  * YOLOX detects people and the ball,
  * Deep-EIoU links detections into tracks using OSNet Re-ID features,
  * a fine-tuned CLIP model predicts, for every tracked box, its role
    (player / goalkeeper / referee / ball / other), its jersey number and its
    jersey color.

Team ids are then assigned globally over the whole half: the two most frequent
player jersey colors define team 0 and team 1, and players with any other color
are assigned to the closest team centroid in Re-ID space.

Output: ``tracks_raw.txt`` (columns: see ``sngsr.utils.TRACK_COLUMNS``).

Usage:
    python -m sngsr.tracking --config configs/baseline.yaml --sequence <name>
"""

import os
import warnings
from pathlib import Path

import clip
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from natsort import natsorted
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchreid.utils import FeatureExtractor
from tqdm import tqdm

from sngsr.jersey.clip_model import CLIPFinetune
from sngsr.utils import TRACK_COLUMNS, allow_reid_checkpoint_globals, get_device, parse_step_args, resolve
from yolox.EIoU_tracker.Deep_EIoU import Deep_EIoU
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.utils.transforms import get_transforms

allow_reid_checkpoint_globals()
warnings.filterwarnings("ignore")

ROLES = {0: "Player", 1: "Goalkeeper", 2: "Referee", 3: "Ball", 4: "Other"}
# Jersey colors the CLIP color head is matched against (also used by the Qwen verifier).
JERSEY_COLORS = ["Red", "Blue", "Green", "Yellow", "White", "Black", "Orange", "Purple", "Pink",
                 "Brown", "Grey", "Charcoal", "Neon yellow"]
DEFAULT_ATTRIBUTES = ("Player", 100, "Unknown")  # 100 = jersey number not visible


##########################################
#   DETECTOR
##########################################

class Predictor:
    """Batched YOLOX inference."""

    def __init__(self, model, exp, device, fp16=False):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    @torch.no_grad()
    def inference_batch(self, imgs):
        """imgs: list of BGR images. Returns the post-processed detections of each image."""
        batch = [preproc(img, self.test_size, self.rgb_means, self.std)[0] for img in imgs]
        batch = torch.from_numpy(np.stack(batch, axis=0)).float().to(self.device)
        if self.fp16:
            batch = batch.half()
        outputs = self.model(batch)
        return postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)


##########################################
#   CLIP ATTRIBUTES
##########################################

def get_number(digit1, digit2):
    """Combine the tens and units heads (class 10 = no digit). 100 means 'not visible'."""
    if digit1 == 10:
        return 100 if digit2 == 10 else digit2
    return digit1 * 10 + digit2


class ClipAttributes:
    """Role, jersey number and jersey color from the fine-tuned CLIP model."""

    def __init__(self, model_path, device):
        self.device = device
        self.model = CLIPFinetune(device=device)
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        self.model = self.model.to(device).eval()
        with torch.inference_mode():
            tokens = clip.tokenize(JERSEY_COLORS).to(device)
            self.color_features = F.normalize(
                self.model.clip_model.encode_text(tokens).float(), p=2, dim=1)

    @torch.inference_mode()
    def __call__(self, images, batch_size=64):
        """images: list of PIL crops. Returns a list of (role, number, color)."""
        results = []
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            x = torch.stack([self.model.preprocess(im) for im in chunk]).to(self.device)
            out = self.model(x)
            roles = out["role_logits"].argmax(1).tolist()
            d1 = out["digit1_logits"].argmax(1).tolist()
            d2 = out["digit2_logits"].argmax(1).tolist()
            emb = F.normalize(out["color_embedding"].float(), p=2, dim=1)
            colors = (emb @ self.color_features.T).argmax(1).tolist()
            results.extend(
                (ROLES[r], get_number(a, b), JERSEY_COLORS[c])
                for r, a, b, c in zip(roles, d1, d2, colors)
            )
        return results


##########################################
#   TEAM ASSIGNMENT
##########################################

def global_color_team_assignment(track_df, row_embs=None):
    """
    1) The two most common player jersey colors of the whole half become
       team 0 and team 1; any other color gets team -1.
    2) If ``row_embs`` (one Re-ID embedding per row, or None) is given, players
       with another color are assigned to the team whose mean embedding is the
       closest (cosine similarity).
    """
    track_df["color"] = track_df["color"].str.strip().str.lower()
    player_mask = track_df["role"].str.lower() == "player"
    if not player_mask.any():
        return track_df

    color_counts = track_df.loc[player_mask, "color"].value_counts()
    if len(color_counts) == 1:
        track_df.loc[player_mask & (track_df["color"] == color_counts.index[0]), "team"] = 0
        return track_df

    color_a, color_b = color_counts.index[0], color_counts.index[1]
    track_df.loc[player_mask & (track_df["color"] == color_a), "team"] = 0
    track_df.loc[player_mask & (track_df["color"] == color_b), "team"] = 1
    leftover_mask = player_mask & ~track_df["color"].isin([color_a, color_b])
    track_df.loc[leftover_mask, "team"] = -1

    if row_embs is not None and len(row_embs) == len(track_df) and leftover_mask.any():
        _assign_leftovers_by_reid(track_df, row_embs, player_mask, leftover_mask)
    return track_df


def _assign_leftovers_by_reid(track_df, row_embs, player_mask, leftover_mask):
    has_emb = np.array([e is not None for e in row_embs])
    teams_col = track_df["team"].to_numpy()

    def centroid(team):
        idx = np.flatnonzero(player_mask.to_numpy() & (teams_col == team) & has_emb)
        if idx.size == 0:
            return None
        c = np.mean([row_embs[i].astype(np.float32) for i in idx], axis=0)
        return c / (np.linalg.norm(c) + 1e-12)

    c0, c1 = centroid(0), centroid(1)
    if c0 is None or c1 is None:
        return
    leftover_idx = np.flatnonzero(leftover_mask.to_numpy() & has_emb)
    if leftover_idx.size == 0:
        return
    E = np.stack([row_embs[i] for i in leftover_idx]).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    track_df.loc[track_df.index[leftover_idx], "team"] = np.where(E @ c0 >= E @ c1, 0, 1)


def unify_team_assignments(track_df):
    """Give every player track a single team: the majority of its 0/1 frames (else -1)."""
    is_player = track_df["role"].str.lower() == "player"
    for tid, group in track_df[is_player].groupby("track_id"):
        assigned = group.loc[group["team"].isin([0, 1]), "team"]
        team = assigned.value_counts().idxmax() if len(assigned) else -1
        track_df.loc[is_player & (track_df["track_id"] == tid), "team"] = team
    return track_df


##########################################
#   PIPELINE
##########################################

class FrameDataset(Dataset):
    def __init__(self, img_dir):
        self.img_paths = natsorted(p for p in os.listdir(img_dir) if p.endswith(".jpg"))
        self.img_paths = [os.path.join(img_dir, p) for p in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        return cv2.imread(path), int(Path(path).stem)


def parse_size(value):
    if isinstance(value, str):
        return tuple(int(v) for v in value.strip("()[] ").split(","))
    return tuple(value)


class TrackingPipeline:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.track_cfg = cfg["TRACKER"]

        # 1) Detector (YOLOX)
        exp = get_exp(resolve(self.track_cfg["EXP_FILE"]), "yolox")
        exp.test_conf = self.track_cfg["CONF"]
        exp.nmsthre = self.track_cfg["NMS_THRESH"]
        exp.test_size = parse_size(self.track_cfg["TSIZE"])
        self.test_size = exp.test_size
        model = exp.get_model().to(device).eval()
        ckpt = torch.load(resolve(self.track_cfg["MODEL_PATH"]), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if self.track_cfg["FP16"]:
            model = model.half()
        self.predictor = Predictor(model, exp, device, fp16=self.track_cfg["FP16"])

        # 2) Re-ID (OSNet)
        self.reid_model = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path=resolve(cfg["REID"]["MODEL_PATH"]),
            device=str(device),
        )
        self.reid_transforms = get_transforms(image_size=tuple(cfg["REID"]["IMAGE_SIZE"]))

        # 3) Role / jersey number / jersey color (CLIP)
        self.attributes = ClipAttributes(resolve(cfg["CLIP"]["MODEL_PATH"]), device)

    @torch.no_grad()
    def process_sequence(self, img_dir, batch_size=4, num_workers=4):
        results = {key: [] for key in TRACK_COLUMNS}
        # One Re-ID embedding per output row (float16 keeps a full half in RAM).
        row_embs = []

        dataset = FrameDataset(img_dir)
        if len(dataset) == 0:
            raise FileNotFoundError(f"No .jpg frame found in {img_dir}")
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                            collate_fn=lambda x: x)
        tracker = Deep_EIoU(self.track_cfg, frame_rate=self.track_cfg["FPS"])

        for batch in tqdm(loader, desc=f"Tracking {os.path.basename(os.path.dirname(img_dir))}"):
            frames, frame_ids = zip(*batch)
            outputs = self.predictor.inference_batch(list(frames))

            for frame, frame_id, output in zip(frames, frame_ids, outputs):
                if output is None or len(output) == 0:
                    continue
                height, width = frame.shape[:2]

                # Detections back to the original image scale
                # NOTE: the scores are rescaled together with the boxes; the
                # tracker thresholds of the baseline config were tuned this way.
                det = output.cpu().numpy()
                det /= min(self.test_size[1] / width, self.test_size[0] / height)
                det = det[~np.any(det[:, 0:4] < 1, axis=1)]
                if len(det) == 0:
                    continue

                # Re-ID features of every detection
                crops = [frame[max(0, int(y1)):min(height, int(y2)), max(0, int(x1)):min(width, int(x2))]
                         for x1, y1, x2, y2, *_ in det]
                tensors = [self.reid_transforms(image=cv2.cvtColor(c, cv2.COLOR_BGR2RGB))["image"]
                           for c in crops]
                embs = self.reid_model(torch.stack(tensors).to(self.device, non_blocking=True))
                embs = F.normalize(embs, p=2, dim=1).cpu().numpy()

                online_targets = tracker.update(det, embs)
                if not online_targets:
                    continue

                # CLIP attributes of every tracked box large enough
                crops_pil, crop_idx = [], []
                for j, t in enumerate(online_targets):
                    x, y, w, h = t.last_tlwh
                    if w * h <= self.track_cfg["MIN_BOX_AREA"]:
                        continue
                    x1, y1 = int(max(0, x)), int(max(0, y))
                    x2, y2 = int(min(width, x + w)), int(min(height, y + h))
                    if x2 > x1 and y2 > y1:
                        crops_pil.append(Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)))
                        crop_idx.append(j)
                predictions = dict(zip(crop_idx, self.attributes(crops_pil))) if crops_pil else {}

                # Team stays -1 here: it is assigned once for the whole half below.
                for j, t in enumerate(online_targets):
                    role, jersey, color = predictions.get(j, DEFAULT_ATTRIBUTES)
                    tlwh = t.last_tlwh
                    row = [frame_id, t.track_id, tlwh[0], tlwh[1], tlwh[2], tlwh[3], t.score,
                           role, jersey, color.strip().lower(), -1]
                    for key, value in zip(TRACK_COLUMNS, row):
                        results[key].append(value)
                    row_embs.append(None if t.curr_feat is None
                                    else np.asarray(t.curr_feat, dtype=np.float16).copy())

        track_df = pd.DataFrame(results)
        track_df = global_color_team_assignment(track_df, row_embs)
        return unify_team_assignments(track_df)


def run(cfg, paths):
    device = get_device(cfg)
    print(f"Using device: {device}")
    if device.type != "cuda" and cfg["TRACKER"].get("FP16"):
        print("WARNING: TRACKER.FP16 needs CUDA, disabled.")
        cfg["TRACKER"]["FP16"] = False
    pipeline = TrackingPipeline(cfg, device)
    track_df = pipeline.process_sequence(paths.img_dir, batch_size=cfg["TRACKER"].get("BATCH_SIZE", 4))
    track_df["frame"] = track_df["frame"].astype(str).str.zfill(6)
    track_df.to_csv(paths.tracks_raw, index=False, header=False)
    print(f"Saved {len(track_df)} tracked boxes to {paths.tracks_raw}")


if __name__ == "__main__":
    _, cfg, paths = parse_step_args("Detection, tracking and CLIP role/jersey/color recognition.")
    run(cfg, paths)
