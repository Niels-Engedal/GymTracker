import os
import cv2
import numpy as np

def overlay_joint_trajectory(
    video_path, df, joint_name, output_path="annotated_video.mp4", frame_rate=30, visualize_velocity="false"
):
    """
    Overlays the 2D trajectory of a joint onto a video, scaling coordinates properly, and optionally visualizes velocity.

    Parameters:
    video_path (str): Path to the input video file (.mov or .mp4).
    df (pd.DataFrame): DataFrame containing trajectory data with 'time', 'X', 'Y', and 'velocity' columns.
    joint_name (str): The name of the joint (e.g., "Hip").
    output_path (str): Path to save the output video (always .mp4). Defaults to "annotated_video.mp4".
    frame_rate (int): Frame rate for the output video. Defaults to 30.
    visualize_velocity (str): How to visualize velocity ('false', 'color', 'thickness', 'both'). Defaults to 'false'.

    Returns:
    None
    """
    print("Starting overlay process...")

    # Validate input file extension
    input_extension = os.path.splitext(video_path)[-1].lower()
    if input_extension not in [".mov", ".mp4"]:
        raise ValueError("Input video must be .mov or .mp4 format.")

    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video properties: width={width}, height={height}, frame_rate={frame_rate}")

    # Prepare output video
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Use H.264 codec
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
    print(f"Output video will be saved at: {output_path}")

    # Extract joint trajectory data
    x_col = f"{joint_name}_X"
    y_col = f"{joint_name}_Y"
    vel_col = f"{joint_name}_velocity"

    print(f"Looking for columns: {x_col}, {y_col}, {vel_col}")
    if x_col not in df.columns or y_col not in df.columns or vel_col not in df.columns:
        print(f"Error: Columns {x_col}, {y_col}, or {vel_col} not found in DataFrame.")
        return

    trajectory = df[[x_col, y_col, vel_col]].dropna()
    print(f"Initial trajectory data (first 5 rows):\n{trajectory.head()}")

    # Normalize velocity for visualization
    velocity_min = trajectory[vel_col].min()
    velocity_max = trajectory[vel_col].max()
    trajectory["velocity_normalized"] = (trajectory[vel_col] - velocity_min) / (velocity_max - velocity_min)
    print(f"Velocity normalized (first 5 rows):\n{trajectory[['velocity_normalized']].head()}")

    # Initialize a blank image to draw the persistent trajectory
    persistent_traj = np.zeros((height, width, 3), dtype=np.uint8)

    prev_x, prev_y = None, None
    frame_idx = 0
    print("Starting video frame processing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break  # End of video

        if frame_idx < len(trajectory):
            x = int(trajectory.iloc[frame_idx][x_col])
            y = int(trajectory.iloc[frame_idx][y_col])
            velocity = trajectory.iloc[frame_idx]["velocity_normalized"]

            # Determine line thickness
            thickness = max(1, int(10 - velocity * 9)) if visualize_velocity in ["thickness", "both"] else 2

            # Determine color based on velocity (Blue-to-Orange Gradient)
            if visualize_velocity in ["color", "both"]:
                blue = np.array([0, 0, 255], dtype=np.uint8)  # Blue in BGR
                orange = np.array([0, 165, 255], dtype=np.uint8)  # Orange in BGR
                color = tuple((blue + (orange - blue) * velocity).astype(int).tolist())  # Convert to tuple of integers
            else:
                color = (0, 255, 0)  # Default green color

            print(f"Frame {frame_idx}: Drawing line with thickness={thickness}, color={color}")

            # Draw trajectory on the persistent image
            if prev_x is not None and prev_y is not None:
                cv2.line(persistent_traj, (prev_x, prev_y), (x, y), color, thickness)

            prev_x, prev_y = x, y

        # Overlay the persistent trajectory on the current frame
        overlay_frame = cv2.addWeighted(frame, 0.8, persistent_traj, 0.7, 0)

        # Write frame to output video
        out.write(overlay_frame)
        frame_idx += 1

    # Release resources
    cap.release()
    out.release()
    print(f"Annotated video saved as MP4 at {output_path}")



def overlay_videos_for_folder(folder_path, merged_df_scaled, joint_name, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through all video files in the folder
    for video_file in os.listdir(folder_path):
        if video_file.endswith(".mov"):  # Adjust if other video formats are included
            try:
                # Parse the ID and video number from the video file name
                # Assuming the video filename follows the format 'id<number>_<number>_baseline.mov'
                parts = video_file.split('_')
                participant_id = parts[0][2:]  # Extract ID after 'id'
                video_number = parts[1]       # Extract video number
                
                # Create video_df dynamically for the current video
                video_df = merged_df_scaled[
                    (merged_df_scaled["participant_id"] == participant_id) &
                    (merged_df_scaled["video_number"] == video_number)
                ].copy()  # Use .copy() to avoid modifying the original DataFrame
                
                if video_df.empty:
                    print(f"No matching data found for {video_file}")
                    continue
                
                # Adjust RAnkle_Y as specified
                #video_df['RAnkle_Y'] += 250
                
                # Construct paths
                video_path = os.path.join(folder_path, video_file)
                output_path = os.path.join(output_folder, f"{video_file.split('.')[0]}_overlay_{joint_name}.mp4")
                
                # Call the overlay function
                overlay_joint_trajectory(
                    video_path=video_path,
                    df=video_df,
                    joint_name=joint_name,
                    output_path=output_path,
                    frame_rate=60  # Adjust if your video has a different frame rate
                )
                print(f"Processed and saved overlay video for {video_file}")
            except Exception as e:
                print(f"Error processing {video_file}: {e}")