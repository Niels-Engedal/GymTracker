import os
import time
import cv2
import pandas as pd
import numpy as np
from Sports2D import Sports2D  # Assuming sports2d is installed
from video_overlay import overlay_joint_trajectory, overlay_videos_for_folder  # Assuming you have this function
from utilities import list_filepaths, load_trc_file, load_mot_file

def get_user_input():
    """Prompt user for participant ID and assigned condition."""
    participant_id = input("Enter Participant ID: ")
    assigned_condition = input("Enter Assigned Condition (baseline, pure, trajectory): ")
    return participant_id, assigned_condition

import os
import cv2
import time

def capture_video(participant_id, video_number, condition, video_dir, duration=10, width=1920, height=1080, frame_rate=60):
    """
    Capture a video using the webcam and save it to the provided directory.
    
    Parameters:
        participant_id (str): ID of the participant.
        video_number (int): Number of the video in the sequence.
        condition (str): Experimental condition.
        video_dir (str): Directory where the video will be saved.
        duration (int): Duration of the video in seconds.
        width (int): Width of the video frame.
        height (int): Height of the video frame.
        frame_rate (int): Frame rate of the video.
        
    Returns:
        str: Full path of the saved video file, or None if capturing failed.
    """
    # Ensure the video directory exists
    os.makedirs(video_dir, exist_ok=True)

    # Construct the full video path
    video_filename = f"{participant_id}_{video_number}_{condition}.mov"
    video_path = os.path.join(video_dir, video_filename)

    # Open the webcam (camera ID 1 for macOS)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Webcam could not be opened.")
        return None

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mov files
    out = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))
    print(f"Recording video: {video_path}")
    start_time = time.time()
    
    # Capture video frames
    while int(time.time() - start_time) < duration:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recording stopped by user.")
                break
        else:
            print("Error capturing frame.")
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video saved: {video_path}")
    return video_path


def process_video_with_sports2d(video_filename, config):
    """Run Sports2D analysis on the video."""
    try:
        config['project']['video_input'] = [video_filename]
        Sports2D.process(config)
    except Exception as e:
        print(f"Error processing video {video_filename}: {e}")

def load_and_combine_trc_mot_data(backflip_data_dir):
    """Load TRC and MOT data, combine, and remove the depth axis."""
    trc_file_paths = list_filepaths(backflip_data_dir, '.trc')
    mot_file_paths = list_filepaths(backflip_data_dir, '.mot')
    print(f"Found {len(trc_file_paths)} TRC files and {len(mot_file_paths)} MOT files.") # DEBUG
    
    # Load the first TRC and MOT files using custom functions
    trc_combined = load_trc_file(trc_file_paths[0])
    mot_combined = load_mot_file(mot_file_paths[0])
    
    # Rename the 'Time' column in the TRC data
    trc_combined = trc_combined.rename(columns={"Time": "time"})

    # Merge TRC and MOT dataframes on shared keys
    merged_df = pd.merge(
        trc_combined, mot_combined,
        on=["time", "participant_id", "video_number", "condition", "person_tracked"],
        how="inner"
    )
    
    # Step 1: Identify remaining axes in the DataFrame
    coordinate_cols = [col for col in merged_df.columns if any(col.endswith(axis) for axis in ["_X", "_Y", "_Z"])]
    joint_names = set(col.rsplit('_', 1)[0] for col in coordinate_cols)  # Extract unique joint names
    remaining_axes = {"X", "Y", "Z"} & set(axis[-1] for axis in coordinate_cols if axis[-2:] in {"_X", "_Y", "_Z"})

    # Calculate variance only for the remaining axes
    variance_results = {axis: 0 for axis in remaining_axes}
    for axis in remaining_axes:
        axis_cols = [f"{joint}_{axis}" for joint in joint_names if f"{joint}_{axis}" in merged_df.columns]
        if axis_cols:  # Only calculate if columns exist
            variance_results[axis] = merged_df[axis_cols].var().mean()

    # Determine the depth axis if multiple remain
    if len(variance_results) > 2:
        depth_axis = min(variance_results, key=variance_results.get)
        print(f"Depth axis identified as: {depth_axis}")

        # Step 2: Remove depth coordinate columns if they exist
        depth_cols = [f"{joint}_{depth_axis}" for joint in joint_names if f"{joint}_{depth_axis}" in merged_df.columns]
        merged_df = merged_df.drop(columns=depth_cols, errors="ignore")
        print(f"Columns removed: {depth_cols}")
    else:
        print(f"Depth axis removal skipped. Remaining axes: {remaining_axes}")
    
    return merged_df


def calculate_derived_metrics(merged_df):
    """Calculate velocities and relative angles."""
    joint_names = set(col.rsplit('_', 1)[0] for col in merged_df.columns if col.endswith('_X'))
    
    # Velocity calculation
    for joint in joint_names:
        for axis in ['X', 'Y']:
            col_name = f"{joint}_{axis}"
            merged_df[f"{joint}_vel_{axis}"] = merged_df[col_name].diff() / merged_df["time"].diff()
        velocity_components = [merged_df[f"{joint}_vel_{axis}"] for axis in ['X', 'Y']]
        merged_df[f"{joint}_velocity"] = np.sqrt(sum(comp**2 for comp in velocity_components))
    
    # Angular velocity
    angular_cols = ["right_knee", "left_knee", "left_ankle", "right_ankle", "left_hip"]
    for col in angular_cols:
        if col in merged_df:
            merged_df[f"{col}_ang_vel"] = merged_df[col].diff() / merged_df["time"].diff()
    
    # Fill NaN
    merged_df = merged_df.fillna(0)
    return merged_df

def calculate_relative_angles(df):
    """Calculate Trunk-Leg and Thigh-Leg angles."""
    df["Trunk_Leg_Angle"] = (df["trunk"] - df["right_thigh"]).abs()
    df["Thigh_Leg_Angle"] = (df["right_thigh"] - df["right_shank"]).abs()
    return df

def scale_coordinates(merged_df):
    """Scale all X and Y coordinates."""
    for col in merged_df.columns:
        if col.endswith("_X"):
            merged_df[col] *= 1000
        elif col.endswith("_Y"):
            merged_df[col] *= -1000
    return merged_df

def main():
    script_dir = os.getcwd()
    video_dir = os.path.normpath(os.path.join(script_dir, "../Videos to Analyze/Webcam_Backflips"))
    participant_id, condition = get_user_input()

    config_path = os.path.normpath(os.path.join(script_dir, "../Configs/backflip_config.toml"))
    print(f"Config file: {config_path}")
    config = config = Sports2D.read_config_file(config_path)

    for video_number in range(1, 6): # change here if you want e.g. 10 videos
        if input("Is the participant ready? (Y/n): ").strip().lower() == 'y':
            video_filename = capture_video(participant_id, video_number, condition, video_dir, duration=4)
            if video_filename:
                process_video_with_sports2d(video_filename, config)
                
                backflip_data_dir = "path_to_backflip_data"  # Set correct path
                merged_df = load_and_combine_trc_mot_data(backflip_data_dir)
                merged_df = calculate_derived_metrics(merged_df)
                merged_df = calculate_relative_angles(merged_df)
                merged_df = scale_coordinates(merged_df)
                
                csv_filename = video_filename.replace(".mov", ".csv")
                merged_df.to_csv(csv_filename, index=False)
                print(f"Data saved to {csv_filename}")
                
                if condition == "trajectory":
                    overlay_videos_for_folder("folder_path", merged_df, "RKnee", "output_folder")
                    
                print(f"Video {video_number} processed.")
                
if __name__ == "__main__":
    main()
