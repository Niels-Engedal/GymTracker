from scipy.signal import find_peaks
import pandas as pd
import numpy as np


from scipy.signal import find_peaks
import pandas as pd
import numpy as np


def analyze_backflip(video_df, peak_alpha=0.05):
    """
    Analyzes a single backflip in a video and calculates preparatory, take-off, grouped, 
    off-grouped, and landing metrics.

    Parameters:
        video_df (pd.DataFrame): Data for a single video containing joint coordinates and angles.

    Returns:
        dict: Metrics for preparatory, take-off, grouped, off-grouped, and landing phases.
    """
    # Define relevant columns
    ankle_x_col, ankle_y_col = "RAnkle_X", "RAnkle_Y"
    trunk_leg_angle_col = "Trunk_Leg_Angle"
    thigh_leg_angle_col = "Thigh_Leg_Angle"
    hip_angle_velocity_col = "left_hip_ang_vel"

    # Ensure the required columns exist
    required_cols = [
        ankle_x_col,
        ankle_y_col,
        trunk_leg_angle_col,
        thigh_leg_angle_col,
        hip_angle_velocity_col,
    ]
    for col in required_cols:
        if col not in video_df.columns:
            raise ValueError(f"Column {col} is missing from the video data.")

    # Detect dips in the ankle Y-coordinate (take-off and landing)
    y_data = video_df[ankle_y_col].values
    time_data = video_df["time"].values

    # Invert the data to find minima
    inverted_y = -y_data
    dips, properties = find_peaks(inverted_y, prominence=peak_alpha)

    # Ensure there is at least one dip on either side of the peak
    peak_height_idx = np.argmax(y_data)
    peak_time = time_data[peak_height_idx]  # Peak time

    # Find dips before and after the peak
    dips_before_peak = dips[dips < peak_height_idx]
    dips_after_peak = dips[dips > peak_height_idx]

    if len(dips_before_peak) == 0 or len(dips_after_peak) == 0:
        raise ValueError("Insufficient dips found before or after the peak height.")

    # Select the largest dip before the peak (take-off) and after the peak (landing)
    take_off_idx = dips_before_peak[np.argmax(inverted_y[dips_before_peak])]
    landing_idx = dips_after_peak[np.argmax(inverted_y[dips_after_peak])]

    # Map relative indices to video_df indices
    take_off_idx = video_df.index[take_off_idx]
    landing_idx = video_df.index[landing_idx]

    # Extract take-off and landing times and coordinates
    take_off_time = video_df.loc[take_off_idx, "time"]
    landing_time = video_df.loc[landing_idx, "time"]
    take_off_x = video_df.loc[take_off_idx, ankle_x_col]
    landing_x = video_df.loc[landing_idx, ankle_x_col]

    # Calculate horizontal displacement
    horizontal_displacement = abs(landing_x - take_off_x)

    # Adjusted Take-off phase end: Determine when the tuck begins or at the peak
    angular_velocity = video_df[thigh_leg_angle_col].diff() / video_df["time"].diff()
    tuck_start_idx = angular_velocity.loc[angular_velocity.index > take_off_idx].idxmin()
    tuck_start_time = video_df.loc[tuck_start_idx, "time"]

    # Take-off end time must be the earlier of tuck start or peak time
    take_off_end_time = min(tuck_start_time, peak_time)

    # Grouped phase: From Take-off end to Off-group start
    grouped_idx = video_df.loc[(video_df["time"] > take_off_end_time) & (video_df.index <= landing_idx)][
        [trunk_leg_angle_col, thigh_leg_angle_col]
    ].mean(axis=1).idxmin()
    grouped_time = video_df.loc[grouped_idx, "time"]

    # Off-grouped Phase Start: Zero-crossing of hip angular velocity
    hip_velocity = video_df[hip_angle_velocity_col]
    hip_velocity_diff = np.sign(hip_velocity).diff()

    # Filter valid zero-crossing points after grouped start
    valid_zero_crossings = hip_velocity_diff.loc[
        (hip_velocity_diff != 0) & (hip_velocity_diff.index > grouped_idx)
    ]

    if valid_zero_crossings.empty:
        off_group_start_idx = grouped_idx
    else:
        off_group_start_idx = valid_zero_crossings.index[0]

    off_group_start_time = video_df.loc[off_group_start_idx, "time"]

    # Define Landing Phase End Time as the last timepoint in the video
    landing_end_time = time_data[-1]

    # Calculate durations
    preparatory_duration = take_off_time - time_data[0]
    take_off_to_grouped_duration = grouped_time - take_off_time
    grouped_to_landing_duration = landing_time - grouped_time
    total_airborne_duration = landing_time - take_off_time

    # Define phases
    phase_intervals = {
        "Preparatory": (time_data[0], take_off_time),
        "Take-off": (take_off_time, take_off_end_time),
        "Grouped": (take_off_end_time, grouped_time),
        "Off-grouped": (grouped_time, landing_time),
        "Landing": (landing_time, landing_end_time),
    }

    # Return calculated metrics
    metrics = {
        "preparatory_duration": preparatory_duration,
        "take_off_time": take_off_time,
        "take_off_x": take_off_x,
        "landing_time": landing_time,
        "landing_x": landing_x,
        "horizontal_displacement": horizontal_displacement,
        "grouped_time": grouped_time,
        "take_off_to_grouped_duration": take_off_to_grouped_duration,
        "grouped_to_landing_duration": grouped_to_landing_duration,
        "total_airborne_duration": total_airborne_duration,
        "phase_intervals": phase_intervals,
    }
    return metrics



def analyze_all_videos(merged_df, peak_alpha=0.05):
    """
    Analyzes all videos in the dataset and calculates metrics per video.

    Parameters:
        merged_df (pd.DataFrame): Combined dataframe containing all videos.

    Returns:
        pd.DataFrame: Metrics for all videos.
    """
    video_metrics = []
    for (participant_id, video_number, condition), video_df in merged_df.groupby(
        ["participant_id", "video_number", "condition"]
    ):
        metrics = analyze_backflip(video_df, peak_alpha=peak_alpha)
        metrics.update(
            {
                "participant_id": participant_id,
                "video_number": video_number,
                "condition": condition,
            }
        )
        video_metrics.append(metrics)

    return pd.DataFrame(video_metrics)
