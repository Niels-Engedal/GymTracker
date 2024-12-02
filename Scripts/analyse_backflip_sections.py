import numpy as np
import pandas as pd

def analyze_backflip(video_df, ground_y_threshold):
    """
    Analyzes a single backflip in a video and calculates take-off, grouped, 
    off-grouped, and landing metrics.

    Parameters:
        video_df (pd.DataFrame): Data for a single video containing joint coordinates and angles.
        ground_y_threshold (float): Y-value threshold to define the ground.

    Returns:
        dict: Metrics for take-off, grouped, off-grouped, and landing.
    """
    # Define relevant columns
    ankle_x_col, ankle_y_col = "RAnkle_X", "RAnkle_Y"
    trunk_leg_angle_col = "Trunk_Leg_Angle"
    thigh_leg_angle_col = "Thigh_Leg_Angle"

    # Ensure the required columns exist
    required_cols = [ankle_x_col, ankle_y_col, trunk_leg_angle_col, thigh_leg_angle_col]
    for col in required_cols:
        if col not in video_df.columns:
            raise ValueError(f"Column {col} is missing from the video data.")

    # Step 1: Detect take-off (ankle leaves ground)
    take_off_idx = video_df[video_df[ankle_y_col] < ground_y_threshold].index.min()
    take_off_time = video_df.loc[take_off_idx, "time"]
    take_off_x = video_df.loc[take_off_idx, ankle_x_col]

    # Step 2: Detect landing (ankle re-enters ground)
    landing_idx = video_df[video_df[ankle_y_col] > ground_y_threshold].index.max()
    landing_time = video_df.loc[landing_idx, "time"]
    landing_x = video_df.loc[landing_idx, ankle_x_col]

    # Step 3: Calculate horizontal displacement
    horizontal_displacement = abs(landing_x - take_off_x)

    # Step 4: Find grouped phase (minimum angles)
    airborne_df = video_df.loc[take_off_idx:landing_idx]  # Only consider airborne phase
    grouped_idx = airborne_df[
        [trunk_leg_angle_col, thigh_leg_angle_col]
    ].mean(axis=1).idxmin()  # Find index of minimum average angle

    grouped_time = video_df.loc[grouped_idx, "time"]

    # Step 5: Calculate durations
    take_off_to_grouped_duration = grouped_time - take_off_time
    grouped_to_landing_duration = landing_time - grouped_time
    total_airborne_duration = landing_time - take_off_time

    # Return the calculated metrics
    metrics = {
        "take_off_time": take_off_time,
        "take_off_x": take_off_x,
        "landing_time": landing_time,
        "landing_x": landing_x,
        "horizontal_displacement": horizontal_displacement,
        "grouped_time": grouped_time,
        "take_off_to_grouped_duration": take_off_to_grouped_duration,
        "grouped_to_landing_duration": grouped_to_landing_duration,
        "total_airborne_duration": total_airborne_duration,
    }
    return metrics

def analyze_all_videos(merged_df, ground_y_threshold):
    """
    Analyzes all videos in the dataset and calculates metrics per video.

    Parameters:
        merged_df (pd.DataFrame): Combined dataframe containing all videos.
        ground_y_threshold (float): Y-value threshold to define the ground.

    Returns:
        pd.DataFrame: Metrics for all videos.
    """
    # Group by video (assuming video identification columns are present)
    video_metrics = []
    for (participant_id, video_number, condition), video_df in merged_df.groupby(
        ["participant_id", "video_number", "condition"]
    ):
        metrics = analyze_backflip(video_df, ground_y_threshold)
        metrics.update({
            "participant_id": participant_id,
            "video_number": video_number,
            "condition": condition,
        })
        video_metrics.append(metrics)

    # Convert to DataFrame for easy analysis
    metrics_df = pd.DataFrame(video_metrics)
    return metrics_df