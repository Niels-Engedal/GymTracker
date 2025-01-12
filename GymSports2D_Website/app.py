from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import subprocess
import os
import time

from Sports2D import Sports2D  # Assuming sports2d is installed

from Scripts.experimental_script import (
    smooth_frequencies,
    generate_waveform,
    sonify_value,
    save_audio,
    combine_audio_and_video,
    process_video_with_sports2d,
    load_and_combine_trc_mot_data,
    scale_coordinates,
    overlay_joint_trajectory,
)



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@app.route('/upload', methods=['POST'])
def upload_and_process():
    file = request.files['video']
    file_path = f"temp_{file.filename}"
    file.save(file_path)

    try:
        # Step 1: Process Video with Sports2D
        config_path = "path/to/config.toml"  # Update with your actual config path
        config = Sports2D.read_config_file(config_path)
        process_video_with_sports2d(file_path, config)

        # Step 2: Load and Scale Trajectory Data
        backflip_data_dir = "path/to/backflip_data_dir"  # Replace with actual path
        merged_df = load_and_combine_trc_mot_data(backflip_data_dir, os.path.basename(file_path))
        merged_df = scale_coordinates(merged_df)

        # Step 3: Overlay Trajectory
        joint_name = "Hip"  # Replace with desired joint
        output_path = f"static/processed_{os.path.splitext(file.filename)[0]}.mp4"
        overlay_joint_trajectory(
            video_path=file_path,
            df=merged_df,
            joint_name=joint_name,
            output_path=output_path,
            frame_rate=30,  # Adjust to match your video
            visualize_velocity="color"
        )

        # Step 4: Clean Up Temporary Files
        os.remove(file_path)

        # Step 5: Serve the Processed Video
        return jsonify({'status': 'Video processed successfully!', 'path': f'/{output_path}'})
    except Exception as e:
        print(f"Error processing video: {e}")
        return jsonify({'status': 'Error processing video', 'error': str(e)})


def cleanup_static_folder():
    static_folder = 'static'
    file_lifetime = 3600  # Time in seconds (1 hour)
    current_time = time.time()

    for filename in os.listdir(static_folder):
        file_path = os.path.join(static_folder, filename)
        if os.path.isfile(file_path):
            # Check file's age
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > file_lifetime:
                os.remove(file_path)
                print(f"Deleted old file: {file_path}")


if __name__ == '__main__':
    app.run(debug=True)
