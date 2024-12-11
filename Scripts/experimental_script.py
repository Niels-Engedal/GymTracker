import os
import time
import cv2
import pandas as pd
import numpy as np
from Sports2D import Sports2D  # Assuming sports2d is installed
from video_overlay import overlay_joint_trajectory, overlay_videos_for_folder  # Assuming you have this function
from utilities import list_filepaths, load_trc_file, load_mot_file, extract_identifiers
import simpleaudio as sa
import moviepy
from scipy.io.wavfile import write
import scipy.signal
import pyinputplus as pyip

# Shared Functions
def smooth_frequencies(frequencies, smooth_factor=5):
    """
    Apply a moving average to smooth frequency transitions.
    """
    kernel = np.ones(smooth_factor) / smooth_factor
    smoothed_frequencies = np.convolve(frequencies, kernel, mode="same")
    return smoothed_frequencies

def generate_waveform(wave_type, frequency, amplitude, t):
    """
    Generate a waveform based on the given type.

    Parameters:
        wave_type (str): Type of waveform ("sine", "triangle", "square", "sawtooth").
        frequency (float): Frequency of the waveform.
        amplitude (float): Amplitude of the waveform.
        t (np.ndarray): Time array for the waveform.

    Returns:
        np.ndarray: Generated waveform.
    """
    if wave_type == "sine":
        return amplitude * np.sin(2 * np.pi * frequency * t)
    elif wave_type == "triangle":
        return amplitude * scipy.signal.sawtooth(2 * np.pi * frequency * t, width=0.5)
    elif wave_type == "square":
        return amplitude * scipy.signal.square(2 * np.pi * frequency * t)
    elif wave_type == "sawtooth":
        return amplitude * scipy.signal.sawtooth(2 * np.pi * frequency * t)
    else:
        raise ValueError("Invalid wave type. Choose 'sine', 'triangle', 'square', or 'sawtooth'.")



def sonify_value(values, frame_rate, mode="height", min_freq=50, max_freq=500, sample_rate=44100, smooth_factor=5, wave_type="triangle"):
    """
    Generate a smooth continuous tone based on input values (height or angle) with optional dynamic loudness.

    Parameters:
        values (np.ndarray): Array of input values (e.g., Y-coordinates or angles).
        frame_rate (int): Frame rate of the video (frames per second).
        mode (str): Either "height" for Y-coordinates or "angle" for angles.
        min_freq (int): Minimum frequency (Hz) for mapping.
        max_freq (int): Maximum frequency (Hz) for mapping.
        sample_rate (int): Sample rate for the audio (Hz).
        smooth_factor (int): Number of frames for frequency smoothing.
        wave_type (str): Type of waveform ("sine", "triangle", "square", "sawtooth").

    Returns:
        np.ndarray: Generated audio waveform.
    """
    base_loudness = 0.1
    dynamic_loudness_coefficient = 0.5


    print(f"DEBUG: Number of input values: {len(values)}")
    print(f"DEBUG: Frame rate: {frame_rate} fps")
    print(f"DEBUG: Sample rate: {sample_rate} Hz")
    print(f"DEBUG: Mode: {mode}")

    # Map values directly to frequency range
    if mode == "height":
        value_to_freq_scale = (max_freq - min_freq) / 1080  # Assuming 1080p height range
        frequencies = max_freq - (values * value_to_freq_scale)
        amplitudes = np.ones_like(values) * base_loudness  # Constant amplitude for height
    elif mode == "angle":
        value_to_freq_scale = (max_freq - min_freq) / 360  # Assuming maximum 360 degrees
        frequencies = min_freq + (values * value_to_freq_scale)
        amplitudes = base_loudness + (1 - values / 360) * dynamic_loudness_coefficient  # Dynamic amplitude: louder for smaller angles
    else:
        raise ValueError("Invalid mode. Use 'height' or 'angle'.")

    print(f"DEBUG: Frequencies range (min, max) before smoothing: {frequencies.min()}, {frequencies.max()}")
    print(f"DEBUG: Amplitudes range (min, max): {amplitudes.min()}, {amplitudes.max()}")

    # Smooth the frequencies
    frequencies = smooth_frequencies(frequencies, smooth_factor=smooth_factor)
    print(f"DEBUG: Frequencies range (min, max) after smoothing: {frequencies.min()}, {frequencies.max()}")

    # Generate a time array for the entire audio duration
    duration = len(values) / frame_rate
    t = np.linspace(0, duration, int(duration * sample_rate), False)

    # Interpolate frequencies and amplitudes to match the audio sample rate
    time_points = np.linspace(0, duration, len(frequencies))
    interpolated_frequencies = np.interp(t, time_points, frequencies)
    interpolated_amplitudes = np.interp(t, time_points, amplitudes)

    # Generate the continuous waveform with dynamic amplitude
    #audio_waveform = interpolated_amplitudes * np.sin(2 * np.pi * interpolated_frequencies * t)
    audio_waveform = generate_waveform(wave_type, interpolated_frequencies, interpolated_amplitudes, t)

    print(f"DEBUG: Audio waveform length (samples): {len(audio_waveform)}")
    return audio_waveform

def save_audio(merged_df, frame_rate, sample_rate, mode="angle", joint_name="Hip", audio_path="debug_audio_output.wav"):
    """
    Generate and save the audio file based on input values (height or angle).

    Parameters:
        merged_df (pd.DataFrame): The processed DataFrame with input data.
        frame_rate (int): Frame rate of the video (frames per second).
        sample_rate (int): Sample rate for the audio (Hz).
        mode (str): Either "height" for Y-coordinates or "angle" for angles.
        joint_name (str): Joint name to sonify (only used in "height" mode).
        audio_path (str): Path to save the audio file.
    """
    if mode == "height":
        if f"{joint_name}_Y" not in merged_df.columns:
            raise KeyError(f"Column '{joint_name}_Y' not found in DataFrame.")
        values = merged_df[f"{joint_name}_Y"].to_numpy()
        print(f"DEBUG: Y-coordinates (first 5): {values[:5]}")
    elif mode == "angle":
        if "Trunk_Leg_Angle" not in merged_df.columns:
            raise KeyError("Column 'Trunk_Leg_Angle' not found in DataFrame.")
        values = merged_df["Trunk_Leg_Angle"].to_numpy()
        print(f"DEBUG: Angles (first 5): {values[:5]}")
    else:
        raise ValueError("Invalid mode. Use 'height' or 'angle'.")

    print(f"DEBUG: Number of values: {len(values)}")
    
    # Generate the audio waveform
    audio_waveform = sonify_value(values, frame_rate, mode=mode, sample_rate=sample_rate, wave_type="triangle")

    # Save the audio waveform to a WAV file
    audio_waveform = (audio_waveform * 32767).astype(np.int16)
    write(audio_path, sample_rate, audio_waveform)
    print(f"DEBUG: Audio file saved to: {audio_path}")



def combine_audio_and_video(video_path, audio_path, output_path):
    """
    Combine a video with an existing audio file.

    Parameters:
        video_path (str): Path to the video file.
        audio_path (str): Path to the audio file.
        output_path (str): Path to save the final video.
    """
    # Load video and audio
    video_clip = moviepy.VideoFileClip(video_path)
    audio_clip = moviepy.audio.io.AudioFileClip.AudioFileClip(audio_path)
    print(f"DEBUG: Video duration: {video_clip.duration} seconds")
    print(f"DEBUG: Audio duration: {audio_clip.duration} seconds")

    # Add audio to the video
    video_with_audio = video_clip.with_audio(audio_clip)

    # Save the final video
    video_with_audio.write_videofile(output_path, codec="libx264", audio_codec="aac")
    print(f"DEBUG: Final video saved to: {output_path}")



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
    print(f"DEBUG: Loading data for video: {video_filename}")
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

    print(f"DEBUG: Combined trajectory data shape: {merged_df.shape}")
    print(f"DEBUG: Trajectory data head:\n{merged_df.head()}")

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
    angular_cols = ["right_knee", "left_knee", "left_ankle", "right_ankle", "left_hip", "right_hip"] # I think we can calculate right_hip as well.
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


def capture_video(participant_id, video_number, condition, video_dir, duration=10, width=1920, height=1080, frame_rate=60):
    """Capture a video using the webcam."""
    os.makedirs(video_dir, exist_ok=True)
    video_filename = f"id{participant_id}_{video_number}_{condition}.mov"
    video_path = os.path.join(video_dir, video_filename)
    cap = cv2.VideoCapture(1) # 1 when not using discord maybe?
    if not cap.isOpened():
        print("Error: Webcam could not be opened.")
        return None
    frames = []
    print(f"Recording video: {video_path}")
    start_time = time.time()
    while int(time.time() - start_time) < duration:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recording stopped by user.")
                break
        else:
            print("Error capturing frame.")
            break
    cap.release()
    cv2.destroyAllWindows()
    frames_to_skip = frame_rate
    trimmed_frames = frames[frames_to_skip:]
    if trimmed_frames:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))
        for frame in trimmed_frames:
            out.write(frame)
        out.release()
        return video_path
    return None


def process_and_save_data(merged_df, output_csv_path):
    """Perform calculations and save the processed data."""
    merged_df = calculate_derived_metrics(merged_df)
    merged_df = calculate_relative_angles(merged_df)
    merged_df = scale_coordinates(merged_df)
    merged_df.to_csv(output_csv_path, index=False)
    print(f"Data saved to {output_csv_path}")
    return merged_df


def process_video(video_filename, config, backflip_data_dir, overlay_dir, joint_to_overlay, frame_rate, visualize_velocity, condition, enable_sonification=False):
    """Process a video and handle analysis, overlays, and outputs."""
    print(f"DEBUG: Starting processing for video: {video_filename}")
    process_video_with_sports2d(video_filename, config)

    # Load and combine trajectory data
    merged_df = load_and_combine_trc_mot_data(backflip_data_dir, video_filename)
    print(f"DEBUG: Loaded trajectory data from {video_filename}")
    print(f"DEBUG: Trajectory data columns: {merged_df.columns}")
    print(f"DEBUG: Trajectory data head:\n{merged_df.head()}")

    likert_options = ['1','2','3','4','5','6']
    
    if condition == "trajectory" or condition == "pure":
        likert_score = pyip.inputMenu(likert_options)
        merged_df["Likert_Score"] = likert_score # saving likert score

    # Save processed data
    csv_filename = os.path.join(backflip_data_dir, os.path.basename(video_filename).replace(".mov", ".csv"))
    merged_df = process_and_save_data(merged_df, csv_filename)

    if condition == "trajectory":
        overlay_path = os.path.join(overlay_dir, f"{os.path.basename(video_filename).replace('.mov', '_overlay.mp4')}")
        print(f"DEBUG: Overlay path: {overlay_path}")

        # Create trajectory overlay
        overlay_joint_trajectory(video_filename, merged_df, joint_to_overlay, overlay_path, frame_rate, visualize_velocity)
        print(f"DEBUG: Overlay video created at {overlay_path}")

        # Check if sonification is enabled
        if enable_sonification:
            final_path = overlay_path.replace("_overlay.mp4", "_final.mp4")

            # Add audio to the overlay
            # Step 1: Save the audio file
            audio_path = overlay_path.replace("_overlay.mp4", "_audio.wav")
            save_audio(merged_df, frame_rate, sample_rate=44100, joint_name=joint_to_overlay, audio_path=audio_path)

            # Step 2: Combine the audio with the video
            combine_audio_and_video(overlay_path, audio_path, final_path)
            print(f"DEBUG: Final video with sonification saved at {final_path}")
            os.system(f"open \"{final_path}\"")
        else:
            print(f"DEBUG: Playing overlay video without sonification: {overlay_path}")
            os.system(f"open \"{overlay_path}\"")

        """overlay_path = os.path.join(overlay_dir, f"{os.path.basename(video_filename).replace('.mov', '_overlay.mp4')}")
        print(f"DEBUG: Overlay path: {overlay_path}")

        # Create trajectory overlay
        overlay_joint_trajectory(video_filename, merged_df, joint_to_overlay, overlay_path, frame_rate, visualize_velocity)
        final_path = overlay_path.replace("_overlay.mp4", "_final.mp4")
        print(f"DEBUG: Final video path: {final_path}")

        # Add audio to the overlay
        # Step 1: Save the audio file
        audio_path = overlay_path.replace("_overlay.mp4", "_audio.wav")
        save_audio(merged_df, frame_rate, sample_rate=44100, joint_name=joint_to_overlay, audio_path=audio_path)

        # Step 2: Combine the audio with the video
        combine_audio_and_video(overlay_path, audio_path, final_path)

        print(f"DEBUG: Playing final video: {final_path}")
        os.system(f"open \"{final_path}\"")"""

    elif condition == "pure":
        print(f"DEBUG: Playing pure video: {video_filename}")
        os.system(f"open {video_filename}")

    elif condition == "baseline":
        print(f"DEBUG: Saved baseline video: {video_filename}")

def main():
    joint_to_overlay = "Hip"
    frame_rate = 30
    num_videos = 1
    duration = 6
    visualize_velocity = "color"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = "/Users/niels/Desktop/University/Third Semester/Perception and Action/Exam/Gymnastics Motion Tracking/Code for Gym Tracking/Videos to Analyze/Webcam_Backflips"
    backflip_data_dir = "/Users/niels/Desktop/University/Third Semester/Perception and Action/Exam/Gymnastics Motion Tracking/Code for Gym Tracking/Analyzed Data/Webcam_Backflip"
    overlay_dir = "/Users/niels/Desktop/University/Third Semester/Perception and Action/Exam/Gymnastics Motion Tracking/Code for Gym Tracking/Overlay Videos"
    os.makedirs(overlay_dir, exist_ok=True)
    config_path = os.path.normpath(os.path.join(script_dir, "../Configs/webcam_backflip_config.toml"))
    config = Sports2D.read_config_file(config_path)

    mode_options = ['Record new videos', 'Analyze existing video']
    condition_options = ['pure', 'trajectory']

    mode = pyip.inputMenu(mode_options, numbered=True)
    if mode == "Analyze existing video":
        video_path = input("Enter path to existing video: ").strip().strip('"').strip("'")
        if not os.path.isfile(video_path):
            print("Invalid file path.")
            return
        condition = pyip.inputMenu(condition_options, numbered=True)
        process_video(video_path, config, backflip_data_dir, overlay_dir, joint_to_overlay, frame_rate, visualize_velocity, condition)
        return

    participant_id = pyip.inputNum("Enter Participant ID: ")
    print(f"Starting {num_videos} baseline videos...")
    for i in range(1, num_videos + 1):
        if pyip.inputYesNo(prompt="Is participant ready? (Yes/No): ") == True:
            video_filename = capture_video(participant_id, i, "baseline", video_dir, duration, frame_rate=frame_rate)
            if video_filename:
                process_video(video_filename, config, backflip_data_dir, overlay_dir, joint_to_overlay, frame_rate, condition="baseline",  visualize_velocity=False,)
                #process_and_save_data(load_and_combine_trc_mot_data(backflip_data_dir, video_filename), os.path.join(backflip_data_dir, f"baseline_{i}.csv"))

    condition = pyip.inputMenu(condition_options, numbered=True)
    if condition not in ['pure', 'trajectory']:
        print("Invalid condition.")
        return
    for i in range(1, num_videos + 1):
        if pyip.inputYesNo(prompt="Is participant ready? (Yes/No): ") == True:
            video_filename = capture_video(participant_id, i, condition, video_dir, duration, frame_rate=frame_rate)
            if video_filename:
                process_video(video_filename, 
                              config, 
                              backflip_data_dir, 
                              overlay_dir, 
                              joint_to_overlay, 
                              frame_rate, 
                              visualize_velocity, 
                              condition, 
                              enable_sonification=False) # edit here if we want sonification


if __name__ == "__main__":
    main()
