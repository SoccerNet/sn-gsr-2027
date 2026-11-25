"""IDATR step 4/4 - project the refined tracks on the pitch.

The bottom-left, bottom-right and bottom-middle points of every box are mapped
with the homography of their frame and expressed in meters from the pitch center.
Output: ``tracks_pitch.txt``.
"""
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sngsr.utils import resolve


def pixel_to_meter_all(x_bottom_left, y_bottom_left, x_bottom_right, y_bottom_right, x_bottom_middle, y_bottom_middle,
                       center_x_px, center_y_px):
    """
    Convert pixel coordinates back to meters.
    Args:
        x_px (float): The x-coordinate in pixels.
        y_px (float): The y-coordinate in pixels.
        center_x_px (float): The x-coordinate of the center of the image in pixels.
        center_y_px (float): The y-coordinate of the center of the image in pixels.

    Returns:
        tuple: The corresponding (x, y) coordinates in meters.
    """
    # Calculate the meter coordinates based on the center
    x_left = x_bottom_left - center_x_px
    y_left = y_bottom_left - center_y_px  # Inverting the y coordinate back
    x_right = x_bottom_right - center_x_px
    y_right = y_bottom_right - center_y_px  # Inverting the y coordinate back
    x_middle = x_bottom_middle - center_x_px
    y_middle = y_bottom_middle - center_y_px  # Inverting the y coordinate back

    return x_left, y_left, x_right, y_right, x_middle, y_middle


def post_process(cfg, paths):
    """Project the refined tracks of one sequence with the per-frame homographies."""
    sfr_cfg = cfg['SFR']

    # Output column names for meter/pixel files
    court_names_meter = [
        "frame", "track_id",
        "x_left", "y_left",
        "x_right", "y_right",
        "x_middle", "y_middle",
        "role", "jersey", "color", "team"
    ]

    # Load the pitch template for coordinate transformations
    template = cv2.imread(resolve(sfr_cfg['TEMPLATE_PATH']))
    center_x_px = template.shape[1] / 2
    center_y_px = template.shape[0] / 2
    for refined_track_path in [paths.tracks_refined]:
        player_position_results_meter = {key: [] for key in court_names_meter}
        data = paths.name

        # Homographies are looked up by frame name (<frame:06d>.npy), not by
        # position in a sorted list: a missing .npy must not shift every
        # following frame onto the wrong homography.
        homo_files = {
            int(Path(p).stem): p
            for p in glob.glob(os.path.join(paths.homography_dir, "*.npy"))
            if Path(p).stem.isdigit()
        }
        homo_cache = {}
        n_missing = 0

        with open(refined_track_path, "r") as refined_track_file:
            for line in refined_track_file:
                # Each line should have 11 columns:
                # frame, track_id, l, t, w, h, conf, role, jersey, color, team
                vals = line.strip().split(",")
                if len(vals) < 11:
                    continue  # skip malformed lines if any

                # Parse numeric fields
                frame_num = int(float(vals[0]))
                tid       = int(vals[1])
                l         = float(vals[2])
                t         = float(vals[3])
                w         = float(vals[4])
                h         = float(vals[5])
                conf      = float(vals[6])
                role   = vals[7]
                jersey = vals[8]
                color  = vals[9]
                team = vals[10]

                # Load the corresponding homography
                if frame_num not in homo_files:
                    n_missing += 1
                    continue
                if frame_num not in homo_cache:
                    homo_cache[frame_num] = np.load(homo_files[frame_num])
                homography = homo_cache[frame_num]

                # Convert to bounding corners
                x_e = l + w
                y_e = t + h
                x_center = l + (w // 2)  # or (w / 2)

                # Build pitch_position array
                x_bottom_left  = l
                y_bottom_left  = y_e
                x_bottom_right = x_e
                y_bottom_right = y_e
                x_bottom_middle = x_center
                y_bottom_middle = y_e

                pitch_position = np.array([
                    [x_bottom_left,  y_bottom_left,  1.],
                    [x_bottom_right, y_bottom_right, 1.],
                    [x_bottom_middle,y_bottom_middle,1.]
                ])
                pitch_position_img = pitch_position.T
                transformed_pitch = np.dot(homography, pitch_position_img)
                player_position_pitch = (transformed_pitch[:2] / transformed_pitch[2]).T

                # Convert to np array
                player_position_pitch = np.array(player_position_pitch)
                # Convert pixel coords to "meter" coords
                x_left, y_left, x_right, y_right, x_middle, y_middle = pixel_to_meter_all(
                    player_position_pitch[0][0], player_position_pitch[0][1],
                    player_position_pitch[1][0], player_position_pitch[1][1],
                    player_position_pitch[2][0], player_position_pitch[2][1],
                    center_x_px, center_y_px
                )

                # Meter-level result
                court_result_meter = [
                    frame_num, tid,
                    x_left, y_left,
                    x_right, y_right,
                    x_middle, y_middle,
                    role, jersey, color, team
                ]
                for k, d in zip(court_names_meter, court_result_meter):
                    player_position_results_meter[k].append(d)

        if n_missing:
            print(f"WARNING: {n_missing} detection(s) skipped in {data}: no homography .npy for their frame")

        # Convert to DataFrames
        meter_df = pd.DataFrame(player_position_results_meter)

        # Build filenames
        meter_filename = paths.tracks_pitch

        print(f"Saving refined track to {meter_filename}")
        meter_df['frame'] = meter_df['frame'].astype(int).astype(str).str.zfill(6)
        meter_df.to_csv(meter_filename, index=False, header=False)



if __name__ == "__main__":
    from sngsr.utils import parse_step_args

    _, cfg, paths = parse_step_args("IDATR 4/4: project the refined tracks on the pitch.")
    post_process(cfg, paths)
