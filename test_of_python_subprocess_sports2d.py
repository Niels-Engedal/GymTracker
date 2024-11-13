import os
from Sports2D import Sports2D

# Define paths
video_directory = "/Users/niels/Desktop/University/Third Semester/Perception and Action/Exam/Gymnastics Motion Tracking/Test Videos/7Nov24/Kraftspring/Normal Speed"
config_path = "/Users/niels/Desktop/University/Third Semester/Perception and Action/Exam/Gymnastics Motion Tracking/Code for Gym Tracking/kraftspring_config_7nov.toml"

# Get a list of all video files in the directory
video_files = [os.path.join(video_directory, f) for f in os.listdir(video_directory) if f.endswith(('.mp4', '.avi', '.mov'))]

# Load the configuration file
config = Sports2D.read_config_file(config_path)

# Update the video input in the config and process each video
for video_file in video_files:
    try:
        config['project']['video_input'] = [video_file]
        Sports2D.process(config)
    except Exception as e:
        print(f"Error processing video {video_file}: {e}")
