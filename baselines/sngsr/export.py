"""
Step 5 - Export the SoccerNet Game State JSON of one match half.

Track-level attributes are obtained by majority voting over each refined
tracklet (role, jersey number, color), the team color is mapped to "left" /
"right" from the mean pitch position of each team over the half, and tracks
outside the pitch, short player / referee tracks and implausible balls are
filtered out.

Output: ``<sequence>.json`` with one prediction per box, in the format scored by
``evaluation/scoring.py`` (GS-HOTA).

Usage:
    python -m sngsr.export --config configs/baseline.yaml --sequence <name>
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sngsr.utils import TRACK_COLUMNS, parse_step_args

SPLIT_CODES = {"train": 1, "valid": 2, "test": 3, "challenge": 4}


# Correct column names for the court meter file.
# (Assuming the file contains 12 comma‐separated values per row.)
court_names_meter = [
    "frame", "track_id", "x_left", "y_left", "x_right", "y_right",
    "x_middle", "y_middle", "role", "jersey", "color", "team"
]

def load_court_meter_data(file_path):
    # Read the file without a header so that our provided names are used.
    df = pd.read_csv(file_path, header=None, names=court_names_meter)
    # Convert numeric columns to numbers
    numeric_cols = ["frame", "track_id", "x_left", "y_left", "x_right", "y_right",
                    "x_middle", "y_middle", "jersey", "team"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    # Case-insensitive labels: colors in lowercase, roles as "Player",
    # "Goalkeeper", ... (the form the comparisons below expect).
    df["color"] = df["color"].astype(str).str.strip().str.lower()
    df["role"] = df["role"].astype(str).str.strip().str.capitalize()
    return df


def majority_jersey(series, threshold=0.97):
    """
    Determine the majority jersey number for a tracklet, with special handling for jersey 100.

    Args:
        series (pd.Series): Series of jersey numbers.
        threshold (float): Proportion threshold to accept 100 as the majority.

    Returns:
        The chosen jersey number (as numeric) or None if no valid votes.
    """
    counts = Counter(series)
    total = sum(counts.values())
    if total == 0:
        return None
    count_100 = counts.get(100, 0)
    prop_100 = count_100 / total
    if prop_100 >= threshold:
        return 100
    else:
        # Exclude jersey 100 from consideration.
        if 100 in counts:
            del counts[100]
        if len(counts) == 0:
            return 100
        return counts.most_common(1)[0][0]

def majority_voting(df):
    # Only consider Player and Goalkeeper for jersey voting
    player_df = df[df["role"].isin(["Player", "Goalkeeper"])]
    # Apply our custom majority_jersey function on each track_id group.
    tracklet_jersey = player_df.groupby("track_id")["jersey"].apply(majority_jersey)
    # For roles, use the most common role overall per track_id.
    tracklet_roles = df.groupby("track_id")["role"].apply(
        lambda x: Counter(x).most_common(1)[0][0]
    )
    tracklet_color = df.groupby("track_id")["color"].apply(
        lambda x: Counter(x).most_common(1)[0][0]
    )
    
    return tracklet_jersey, tracklet_roles, tracklet_color

def determine_team_sides(df):
    from collections import Counter
    # -----------------------
    # STEP 1: Assign players using your original approach
    # -----------------------
    valid_df = df[df.team.isin([0, 1])]
    print(valid_df.groupby(['role', 'color']).size().reset_index(name='count').sort_values(by='count', ascending=False))
    if not valid_df.empty:
        team_a = valid_df[valid_df.team == 0]
        team_b = valid_df[valid_df.team == 1]
        avg_a = np.nanmean(team_a["x_middle"])
        avg_b = np.nanmean(team_b["x_middle"])
        # Map team based on which group has the lower average x_middle
        if avg_a < avg_b:
            mapping = {0: "left", 1: "right"}
        else:
            mapping = {0: "right", 1: "left"}

        threshold = (avg_a + avg_b) / 2
    else:
        threshold = np.nanmedian(df["x_middle"])
        mapping = {}

    def assign_team_side(row):
        # If this detection has a valid team value, use the mapping
        if row["team"] in mapping:
            return mapping[row["team"]]
        else:
            # Otherwise compare x_middle to threshold
            return "left" if row["x_middle"] < threshold else "right"

    # Apply to all rows initially
    df["team_side"] = df.apply(assign_team_side, axis=1)

    # -----------------------
    # STEP 2: Override logic for goalkeepers
    # -----------------------
    # If a goalkeeper's x_middle is > 0, force it to "right", else "left"
    # (assuming your pitch center is x=0).
    # This is a simple approach for keepers who stand near their respective goals.
    is_gk = (df["role"] == "Goalkeeper") & (~df["x_middle"].isna())
    df.loc[is_gk & (df["x_middle"] > 0), "team_side"] = "right"
    df.loc[is_gk & (df["x_middle"] <= 0), "team_side"] = "left"

    # -----------------------
    # STEP 3: Majority team side per tracklet
    # -----------------------
    tracklet_team_side = (
        df[df["role"].isin(["Player", "Goalkeeper"])]
        .groupby("track_id")["team_side"]
        .apply(lambda x: Counter(x.dropna()).most_common(1)[0][0] if len(x.dropna()) > 0 else None)
    )

    return tracklet_team_side


def generate_tracklet_json(tracklet_jersey, tracklet_roles, tracklet_team_side, tracklet_color):
    # Allowed roles in final JSON
    allowed_roles = {"other", "player", "goalkeeper", "referee", "ball"}
    tracklet_results = {}
    for track_id in tracklet_roles.index:
        # Convert role to lowercase.
        role = str(tracklet_roles[track_id]).lower()
        if role not in allowed_roles:
            role = "other"

        # For ball and referee, jersey and team are set to None.
        if role in ["ball", "referee"]:
            tracklet_results[str(track_id)] = {"role": role, "jersey": None, "team": None, "color": None}
        else:
            jersey_val = tracklet_jersey[track_id] if track_id in tracklet_jersey.index else None
            jersey_str = str(jersey_val) if jersey_val is not None else None
            if jersey_str == "100":
                jersey_str = None
            team = tracklet_team_side.get(track_id, None)
            color = tracklet_color[track_id] if track_id in tracklet_color.index else None
            tracklet_results[str(track_id)] = {
                "role": role,
                "jersey": jersey_str,
                "team": team,
                "color": color
            }
    return tracklet_results


def postprocess_ball_only(predictions, max_ball_area=800):
    """
    For each track_id, compute the average bounding-box area across all frames.
    If the average area is below 'max_ball_area', override the entire track's role to 'ball'.
    Otherwise, keep whatever role(s) the pipeline assigned.

    Args:
        predictions (list of dict): The final detection list from write_predictions_json.
        max_ball_area (float): Threshold below which we force the entire track to 'ball'.

    Returns:
        list of dict: Updated predictions with track-level overrides for ball only.
    """
    from collections import defaultdict

    # 1) Gather bounding-box areas per track
    track_areas = defaultdict(list)
    for det in predictions:
        track_id = det["track_id"]
        w = det["bbox_image"]["w"]
        h = det["bbox_image"]["h"]
        area = w * h
        track_areas[track_id].append(area)

    # 2) Decide if the track is 'ball' or not
    #    If average area < max_ball_area => force 'ball'
    #    Else => do not override
    track_is_ball = {}
    for track_id, areas in track_areas.items():
        avg_area = sum(areas) / len(areas)
        if avg_area < max_ball_area:
            track_is_ball[track_id] = True
        else:
            track_is_ball[track_id] = False

    # 3) Override role to 'ball' only if track_is_ball[track_id] is True
    for det in predictions:
        track_id = det["track_id"]
        if track_is_ball[track_id]:
            det["attributes"]["role"] = "ball"

    return predictions

# Define the structure for the prediction
def get_most_frequent_color_by_team(predictions):
    # Dictionary to store the color counts for each team
    color_count = defaultdict(int)
    color_count_lr = {
        "left": defaultdict(int),
        "right": defaultdict(int)
    }
    team_count = defaultdict(int)

    # Iterate through each prediction and count the colors based on the team
    for prediction in predictions:
        team = prediction['attributes']['team']
        color = prediction['attributes']['color']
        color_count[color] += 1
        if team == "left" or team == "right":
            color_count_lr[team][color] += 1
        team_count[team] += 1

    def get_most_frequent_color(team):
        most_frequent_color = ""
        max_count = 0
        for color, count in color_count_lr[team].items():
            if count > max_count:
                most_frequent_color = color
                max_count = count
        return most_frequent_color
    
    most_frequent_left_color = get_most_frequent_color("left")
    most_frequent_right_color = get_most_frequent_color("right")

    # Sort the color counts and team counts to determine the most frequent
    sorted_colors = sorted(
        ((color, count) for color, count in color_count.items() if color is not None),
        key=lambda x: x[1],
        reverse=True
    )

    sorted_teams = sorted(
        ((team, count) for team, count in team_count.items() if team is not None),
        key=lambda x: x[1],
        reverse=True
    )
    print(sorted_colors, sorted_teams)
    # Identify the most frequent color and the second and third most frequent
    most_frequent_color = sorted_colors[0][0] if sorted_colors else None

    second_most_frequent_color = sorted_colors[1][0] if len(sorted_colors) > 1 else None
    third_most_frequent_color = sorted_colors[2][0] if len(sorted_colors) > 2 else None

    most_frequent_team = sorted_teams[0][0] if sorted_teams else None
    second_most_frequent_team = sorted_teams[1][0] if len(sorted_teams) > 1 else None

    # Iterate through the predictions and update the team based on the conditions
    updated_predictions = []
    for prediction in predictions:
        team = prediction['attributes']['team']
        color = prediction['attributes']['color']
        role = prediction['attributes']['role']

        # If role is "player", we need to check conditions to change the team
        if role == "player":
            if team == "left" and color != most_frequent_left_color:
                # Change left to right and vice versa
                prediction['attributes']['team'] = "right"
            elif team == "right" and color != most_frequent_right_color:
                prediction['attributes']['team'] = "left"

            # If color is the third most frequent, change the team to second most frequent team
            if color == third_most_frequent_color:
                if second_most_frequent_team:
                    print(color, second_most_frequent_team)
                    prediction['attributes']['team'] = second_most_frequent_team

        # Add the updated prediction to the list
        updated_predictions.append(prediction)
    
    return updated_predictions

# 4) Filter out referees who are far outside the pitch
def is_inside_pitch(bbox_pitch, half_width=52.5, half_height=34, margin_x=10, margin_y=5):
    min_x = -half_width - margin_x
    max_x = half_width + margin_x
    min_y = -half_height - margin_y
    max_y = half_height + margin_y
    x_mid = bbox_pitch["x_bottom_middle"]
    y_mid = bbox_pitch["y_bottom_middle"]
    return (min_x <= x_mid <= max_x) and (min_y <= y_mid <= max_y)

def build_image_ids(img_dir, split, sequence):
    """Map frame number -> SoccerNet image_id.

    The image ids of ``Labels-GameState.json`` are used when the file is present
    next to ``img1/`` (they must match exactly, otherwise the evaluation silently
    ignores the predictions). Otherwise the SoccerNet convention
    ``<split code><sequence number><frame:06d>`` is used.
    """
    labels = os.path.join(os.path.dirname(img_dir), "Labels-GameState.json")
    if os.path.isfile(labels):
        with open(labels) as f:
            images = json.load(f)["images"]
        mapping = {int(os.path.splitext(im["file_name"])[0]): str(im["image_id"]) for im in images}
        print(f"Using the image ids of {labels}")
        return lambda frame: mapping.get(frame)
    code = SPLIT_CODES.get(split, 3)
    seq_id = sequence.split("-")[-1]
    return lambda frame: f"{code}{seq_id}{frame:06d}"


def write_predictions_json(tracklet_results, court_meter_df, sngs_df, output_json_path, video_id, image_id_of):
    predictions = []
    # Index of the pitch positions: (track_id, frame) -> first row, and
    # track_id -> first row as a fallback (O(N) instead of one scan per box).
    by_track_frame, by_track = {}, {}
    for row in court_meter_df.itertuples(index=False):
        by_track_frame.setdefault((int(row.track_id), int(row.frame)), row)
        by_track.setdefault(int(row.track_id), row)

    # Use enumerate to get an incremental id for each detection (line) in the tracking file.
    for i, image_data in enumerate(sngs_df.itertuples(index=False)):
        track_id = int(image_data.track_id)
        frame_num = int(image_data.frame)
        track_data = by_track_frame.get((track_id, frame_num), by_track.get(track_id))
        if track_data is None:
            continue
        image_id = image_id_of(frame_num)
        if image_id is None:  # frame not listed in Labels-GameState.json
            continue

        # Get the track-level attributes if available, otherwise use defaults.
        if str(track_id) in tracklet_results:
            track_info = tracklet_results[str(track_id)]
        else:
            track_info = {"role": "other", "jersey": None, "team": None, "color": None}

        prediction = {
            "bbox_pitch": {
                "x_bottom_left": round(float(track_data.x_left), 3),
                "y_bottom_left": round(float(track_data.y_left), 3),
                "x_bottom_right": round(float(track_data.x_right), 3),
                "y_bottom_right": round(float(track_data.y_right), 3),
                "x_bottom_middle": round(float(track_data.x_middle), 3),
                "y_bottom_middle": round(float(track_data.y_middle), 3)
            },
            "bbox_image": {
                "x": round(float(image_data.x), 2),
                "y": round(float(image_data.y), 2),
                "w": round(float(image_data.w), 2),
                "h": round(float(image_data.h), 2)
            },
            "category_id": 1.0,
            "image_id": image_id,
            "frame": frame_num,
            "video_id": video_id,
            "track_id": track_id,
            "supercategory": "object",
            "attributes": {
                "role": track_info["role"],
                "jersey": track_info["jersey"],
                "team": track_info["team"],
                "color": track_info["color"]
            },
            "id": str(i)  # Incremental id starting from 0 (line number in tracking result file)
        }
        predictions.append(prediction)

    # Now apply the postprocessing step:
    predictions = postprocess_ball_only(predictions, max_ball_area=800)

    predictions = get_most_frequent_color_by_team(predictions)

    # --- Filtering Step ---
    # Remove predictions with role "other"
    filtered_predictions = [pred for pred in predictions if pred["attributes"]["role"] != "other"]

    # --- Filtering Step 2: Remove short tracklets for players and referee only.
    # Compute the number of detections per track_id.
    min_frame_threshold = 20
    track_frame_counts = defaultdict(int)
    for pred in filtered_predictions:
        track_frame_counts[pred["track_id"]] += 1

    # For tracklets whose role is "player" or "referee", only keep them if they have at least min_frame_threshold detections.
    filtered_predictions = [
        pred for pred in filtered_predictions
        if (pred["attributes"]["role"].lower() not in ["player", "referee"])
           or (track_frame_counts[pred["track_id"]] >= min_frame_threshold)
    ]

    # Optionally, remove predictions with role "ball" if remove_ball is True
    # if remove_ball:
    #     filtered_predictions = [pred for pred in filtered_predictions if pred["attributes"]["role"] != "ball"]

    final_filtered_predictions = [
        pred for pred in filtered_predictions if is_inside_pitch(pred["bbox_pitch"])
    ]

    # --- Remove the "color" attribute before writing JSON ---
    for pred in final_filtered_predictions:
        if "color" in pred["attributes"]:
            del pred["attributes"]["color"]

    # 5) Write JSON
    # Compact JSON: a full half holds ~1M boxes.
    final_output = {"predictions": final_filtered_predictions}
    with open(output_json_path, 'w') as f:
        json.dump(final_output, f, separators=(",", ":"))
    print(f"Predictions have been written to {output_json_path}")

def run(cfg, paths):
    court_meter_df = load_court_meter_data(paths.tracks_pitch)
    sngs_df = pd.read_csv(paths.tracks_refined, header=None, names=TRACK_COLUMNS)
    sngs_df["track_id"] = pd.to_numeric(sngs_df["track_id"], errors="coerce")

    # Track-level attributes by majority voting, team side (left/right) per tracklet.
    tracklet_jersey, tracklet_roles, tracklet_color = majority_voting(court_meter_df)
    tracklet_team_side = determine_team_sides(court_meter_df)
    tracklet_results = generate_tracklet_json(tracklet_jersey, tracklet_roles, tracklet_team_side, tracklet_color)

    video_id = paths.name.split("-")[-1]
    image_id_of = build_image_ids(paths.img_dir, cfg["SPLIT"], paths.name)
    write_predictions_json(tracklet_results, court_meter_df, sngs_df, paths.predictions, video_id, image_id_of)


if __name__ == "__main__":
    _, cfg, paths = parse_step_args("Export the SoccerNet Game State JSON.")
    run(cfg, paths)
