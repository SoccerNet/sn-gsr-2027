"""IDATR step 2/4 - build tracklets with one OSNet Re-ID feature per detection."""
import os
import pickle

import numpy as np
import torch
import torchvision.transforms as T
from loguru import logger
from PIL import Image
from torchreid.utils import FeatureExtractor
from tqdm import tqdm

from sngsr.idatr.tracklet import Tracklet
from sngsr.utils import allow_reid_checkpoint_globals, get_device, resolve

allow_reid_checkpoint_globals()

def generate_tracklets(cfg, paths, batch_size=256):
    device = str(get_device(cfg))
    extractor = FeatureExtractor(
        model_name="osnet_x1_0",
        model_path=resolve(cfg["IDATR"]["MODEL_PATH"]),
        device=device,
    )
    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    track_res = np.genfromtxt(paths.tracks_dedup, dtype=str, delimiter=",", encoding="utf-8", ndmin=2)
    frame_ids = track_res[:, 0].astype(int)
    # Group rows by frame once instead of scanning the whole half for every frame.
    order = np.argsort(frame_ids, kind="stable")
    frames, starts = np.unique(frame_ids[order], return_index=True)
    bounds = list(starts[1:]) + [len(order)]

    seq_tracks = {}
    for frame_id, start, end in tqdm(zip(frames, starts, bounds), total=len(frames),
                                     desc=f"Tracklets {paths.name}"):
        img = Image.open(os.path.join(paths.img_dir, f"{frame_id:06d}.jpg")).convert("RGB")
        crops, tids = [], []
        for row in track_res[order[start:end]]:
            frame, track_id, l, t, w, h, score, role, jn, jc, team = row
            frame, track_id = int(frame), int(track_id)
            l, t, w, h, score = float(l), float(t), float(w), float(h), float(score)
            bbox = [l, t, w, h]
            if track_id not in seq_tracks:
                seq_tracks[track_id] = Tracklet(track_id, frame, score, bbox, role, jn, jc, team)
            else:
                seq_tracks[track_id].append_det(frame, score, bbox, role, jn, jc, team)
            crops.append(val_transforms(img.crop((l, t, l + w, t + h))))
            tids.append(track_id)

        feats = []
        for b in range(0, len(crops), batch_size):
            feats.append(extractor(torch.stack(crops[b:b + batch_size])).cpu().numpy())
        feats = np.concatenate(feats) if feats else []
        for tid, feat in zip(tids, feats):
            seq_tracks[tid].append_feat(feat / np.linalg.norm(feat))

    with open(paths.tracklets, "wb") as f:
        pickle.dump(seq_tracks, f)
    logger.info(f"Saved {len(seq_tracks)} tracklets to {paths.tracklets}")


if __name__ == "__main__":
    from sngsr.utils import parse_step_args

    _, cfg, paths = parse_step_args("IDATR 2/4: build tracklets with Re-ID features.")
    generate_tracklets(cfg, paths)
