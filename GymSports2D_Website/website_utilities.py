import os
import time
from Sports2D import Sports2D

def process_video_for_website(video_filename, config, user_dir):
    """
    Run Sports2D analysis on the video and ensure outputs are saved to user_dir.
    
    Parameters:
        video_filename (str): Path to the video to process.
        config (dict): Sports2D configuration dictionary.
        user_dir (str): Directory where the results should be saved.
    """
    try:
        # Dynamically update the result_dir in the config
        config['project']['result_dir'] = user_dir  # Set the output directory for results
        config['project']['video_input'] = [video_filename]  # Set the video input

        # Process the video with Sports2D
        Sports2D.process(config)
    except Exception as e:
        print(f"Error processing video {video_filename}: {e}")
        raise


def get_user_data_dir(user_id):
    """
    Get or create a directory for a specific user's data.

    Parameters:
        user_id (str): The unique identifier for the user.

    Returns:
        str: Path to the user's data directory.
    """
    user_data_dir = os.path.join("static", "data", user_id)
    os.makedirs(user_data_dir, exist_ok=True)
    return user_data_dir


def get_user_processed_dir(user_id):
    """
    Get or create a directory for a specific user's processed videos.

    Parameters:
        user_id (str): The unique identifier for the user.

    Returns:
        str: Path to the user's processed directory.
    """
    user_processed_dir = os.path.join("static", "processed", user_id)
    os.makedirs(user_processed_dir, exist_ok=True)
    return user_processed_dir


def cleanup_static_folder(static_folder='static', file_lifetime=3600):
    """
    Periodically clean up old files from the static folder.

    Parameters:
        static_folder (str): Path to the static folder.
        file_lifetime (int): Lifetime of files in seconds (default: 1 hour).
    """
    current_time = time.time()

    for root, _, files in os.walk(static_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                # Check file's age
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > file_lifetime:
                    os.remove(file_path)
                    print(f"Deleted old file: {file_path}")
