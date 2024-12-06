import os
import time
import cv2
import pandas as pd
import numpy as np
from Sports2D import Sports2D  # Assuming sports2d is installed
from video_overlay import overlay_joint_trajectory, overlay_videos_for_folder  # Assuming you have this function
from utilities import list_filepaths, load_trc_file, load_mot_file, extract_identifiers

def get_user_input():
    """Prompt user for participant ID and assigned condition."""
    participant_id = input("Enter Participant ID: ")
    assigned_condition = input("Enter Assigned Condition (baseline, pure, trajectory): ")
    return participant_id, assigned_condition

def capture_video(participant_id, video_number, condition, video_dir, duration=10, width=1920, height=1080, frame_rate=60):
    """
    Capture a video using the webcam and save it to the provided directory after cutting the first second.
    
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
    video_filename = f"id{participant_id}_{video_number}_{condition}.mov"
    video_path = os.path.join(video_dir, video_filename)

    # Open the webcam (camera ID 1 for macOS)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Webcam could not be opened.")
        return None

    # Temporary storage for frames
    frames = []
    print(f"Recording video: {video_path}")
    start_time = time.time()
    
    # Capture video frames
    while int(time.time() - start_time) < duration:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)  # Store frames in memory
            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recording stopped by user.")
                break
        else:
            print("Error capturing frame.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Remove the first second of frames
    frames_to_skip = frame_rate  # Number of frames to skip (1 second of footage)
    trimmed_frames = frames[frames_to_skip:]

    # Save the trimmed video
    if trimmed_frames:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mov files
        out = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

        for frame in trimmed_frames:
            out.write(frame)
        out.release()
        print(f"Trimmed video saved: {video_path}")
        return video_path
    else:
        print("Error: Not enough frames to save after trimming.")
        return None



def process_video_with_sports2d(video_filename, config):
    """Run Sports2D analysis on the video."""
    try:
        config['project']['video_input'] = [video_filename]
        Sports2D.process(config)
    except Exception as e:
        print(f"Error processing video {video_filename}: {e}")

def load_and_combine_trc_mot_data(backflip_data_dir, video_filename):
    """
    Load TRC and MOT data for a specific video, combine, and remove the depth axis.
    
    Parameters:
        backflip_data_dir (str): Path to the directory containing .trc and .mot files.
        video_filename (str): Filename of the captured video (e.g., "id1_1_baseline.mov").
    
    Returns:
        pd.DataFrame: Combined DataFrame with depth axis removed.
    """
    # Extract metadata from the video filename
    participant_id, video_number, condition, _ = extract_identifiers(video_filename)

    # Find all TRC and MOT files in the directory
    trc_file_paths = list_filepaths(backflip_data_dir, '.trc')
    mot_file_paths = list_filepaths(backflip_data_dir, '.mot')

    # Filter files for the specific video
    trc_file = next(
        (path for path in trc_file_paths if f"id{participant_id}_{video_number}_{condition}" in path), None
    )
    mot_file = next(
        (path for path in mot_file_paths if f"id{participant_id}_{video_number}_{condition}" in path), None
    )

    if not trc_file or not mot_file:
        raise FileNotFoundError(f"No TRC or MOT files found for video: {video_filename}")

    # Load the TRC and MOT files
    _, trc_combined = load_trc_file(trc_file)  # Extract the DataFrame from the tuple
    _, mot_combined = load_mot_file(mot_file)  # Extract the DataFrame from the tuple

    # Add metadata to the DataFrames
    trc_combined['participant_id'] = participant_id
    trc_combined['video_number'] = video_number
    trc_combined['condition'] = condition
    trc_combined['person_tracked'] = "person0"  # Default value (adjust if needed)

    mot_combined['participant_id'] = participant_id
    mot_combined['video_number'] = video_number
    mot_combined['condition'] = condition
    mot_combined['person_tracked'] = "person0"  # Default value (adjust if needed)

    # Rename the 'Time' column in the TRC data
    if "Time" in trc_combined.columns:
        trc_combined = trc_combined.rename(columns={"Time": "time"})
    else:
        raise KeyError("'Time' column is missing in the TRC file.")

    # Merge TRC and MOT dataframes on shared keys
    merged_df = pd.merge(
        trc_combined,
        mot_combined,
        on=["time", "participant_id", "video_number", "condition", "person_tracked"],
        how="inner"
    )

    # Remove the Z-axis
    z_cols = [col for col in merged_df.columns if col.endswith("_Z")]
    merged_df = merged_df.drop(columns=z_cols, errors="ignore")
    print(f"Z-axis columns removed: {z_cols}")

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
    joint_to_overlay = "Hip"  # Joint to overlay on the video
    frame_rate = 30
    num_videos = 10  # Number of videos per condition
    duration = 6  # Duration of each video in seconds
    visualize_velocity = "color"  # Options: "thickness", "color", "both"

    # Set paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = "/Users/niels/Desktop/University/Third Semester/Perception and Action/Exam/Gymnastics Motion Tracking/Code for Gym Tracking/Videos to Analyze/Webcam_Backflips"
    backflip_data_dir = "/Users/niels/Desktop/University/Third Semester/Perception and Action/Exam/Gymnastics Motion Tracking/Code for Gym Tracking/Analyzed Data/Webcam_Backflip"
    overlay_dir = "/Users/niels/Desktop/University/Third Semester/Perception and Action/Exam/Gymnastics Motion Tracking/Code for Gym Tracking/Overlay Videos"
    os.makedirs(overlay_dir, exist_ok=True)  # Ensure the overlay directory exists

    #print(f"Video directory: {video_dir}")
    #print(f"Backflip data directory: {backflip_data_dir}")
    #print(f"Overlay directory: {overlay_dir}")

    # Load the configuration file
    config_path = os.path.normpath(os.path.join(script_dir, "../Configs/webcam_backflip_config.toml"))
    #print(f"Config file: {config_path}")
    config = Sports2D.read_config_file(config_path)

    # Get user input
    participant_id = input("Enter Participant ID: ")
    condition = input("Enter Condition (pure or trajectory): ").strip().lower()
    if condition not in ["pure", "trajectory"]:
        print("Invalid condition. Please enter 'pure' or 'trajectory'.")
        return

    # Video number starts at 1 and increments sequentially
    video_number = 1

    """# Record and process baseline videos
    print()
    print(f"Starting {num_videos} baseline videos...")
    print()
    for _ in range(num_videos):
        if input("Is the participant ready? (Y/n): ").strip().lower() == 'y':
            print(f"Recording baseline video {video_number}...")
            print()
            video_filename = capture_video(participant_id, video_number, "baseline", video_dir, duration=duration, frame_rate=frame_rate)
            if video_filename:
                process_video_with_sports2d(video_filename, config)
                try:
                    merged_df = load_and_combine_trc_mot_data(backflip_data_dir, video_filename)
                except FileNotFoundError as e:
                    print(f"Error processing baseline video {video_number}: {e}")
                    video_number += 1
                    continue

                # Perform calculations on the data
                merged_df = calculate_derived_metrics(merged_df)
                merged_df = calculate_relative_angles(merged_df)
                merged_df = scale_coordinates(merged_df)

                # Save the combined data as CSV
                csv_filename = os.path.join(backflip_data_dir, os.path.basename(video_filename).replace(".mov", ".csv"))
                merged_df.to_csv(csv_filename, index=False)
                print(f"Baseline data saved to {csv_filename}")
                print()
            video_number += 1"""

    # Record and process videos based on condition
    print(f"Starting {num_videos} {condition} videos...")
    for _ in range(num_videos):
        if input("Is the participant ready? (Y/n): ").strip().lower() == 'y':
            print(f"Recording {condition} video {video_number}...")
            print()
            video_filename = capture_video(participant_id, video_number, condition, video_dir, duration=duration, frame_rate=frame_rate)
            if video_filename:
                process_video_with_sports2d(video_filename, config)
                try:
                    merged_df = load_and_combine_trc_mot_data(backflip_data_dir, video_filename)
                except FileNotFoundError as e:
                    print(f"Error processing {condition} video {video_number}: {e}")
                    video_number += 1
                    continue

                # Perform calculations on the data
                merged_df = calculate_derived_metrics(merged_df)
                merged_df = calculate_relative_angles(merged_df)
                merged_df = scale_coordinates(merged_df)

                # Save the combined data as CSV
                csv_filename = os.path.join(backflip_data_dir, os.path.basename(video_filename).replace(".mov", ".csv"))
                merged_df.to_csv(csv_filename, index=False)
                print(f"{condition.capitalize()} data saved to {csv_filename}")
                print()

                # Generate overlay or display video based on condition
                if condition == "trajectory":
                    overlay_path = os.path.join(overlay_dir, f"{os.path.basename(video_filename).replace('.mov', '_overlay.mp4')}")
                    overlay_joint_trajectory(
                        video_path=video_filename,
                        df=merged_df,
                        joint_name=joint_to_overlay,
                        output_path=overlay_path,
                        frame_rate=frame_rate,
                        visualize_velocity=visualize_velocity
                    )
                    print(f"Overlay saved to: {overlay_path}")
                    print()
                    print("Displaying overlay...")
                    print()
                    os.system(f"open \"{overlay_path}\"")  # Ensure the path is quoted for spaces
                elif condition == "pure":
                    print("Displaying video...")
                    print()
                    os.system(f"open {video_filename}")  # macOS; adjust for Windows/Linux
            video_number += 1
                
if __name__ == "__main__":
    main()
