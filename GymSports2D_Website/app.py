from flask import Flask, request, render_template, jsonify
import os
import sys
import time

# Import website utilities
from website_utilities import get_user_data_dir, get_user_processed_dir, cleanup_static_folder, process_video_for_website
from Sports2D import Sports2D  # Assuming sports2d is installed

# Dynamically add the Scripts folder to the Python path
scripts_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Scripts"))
print("Scripts folder being added to sys.path:", scripts_folder)  # Debugging line
if scripts_folder not in sys.path:
    sys.path.append(scripts_folder)

# Import experimental script functions
from experimental_script import (
    process_video_with_sports2d,
    load_and_combine_trc_mot_data,
    scale_coordinates,
    overlay_joint_trajectory,
)

# Get the root directory of the project (Code for Gym Tracking)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Construct the path to the Configs folder
configs_folder = os.path.join(project_root, "Configs")
config_path = os.path.join(configs_folder, "website_backflip_config.toml")

print("Config path being used:", config_path)  # Debugging line

# Define data_dir to store all user-related data
data_dir = os.path.join(project_root, "static", "data")
os.makedirs(data_dir, exist_ok=True)  # Ensure the directory exists

# Use this path in your Sports2D config
config = Sports2D.read_config_file(config_path)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_and_process():
    file = request.files['video']

    # Dynamically create a user-specific directory within data_dir
    user_dir = os.path.join(data_dir, "test_user")
    os.makedirs(user_dir, exist_ok=True)  # Ensure user-specific directory exists
    file_path = os.path.join(user_dir, file.filename)
    file.save(file_path)

    try:
        # Step 1: Load and Update Config Dynamically
        config = Sports2D.read_config_file(config_path)

        # Step 2: Process Video with Sports2D
        process_video_with_sports2d(file_path, config, user_dir)

        # Step 3: Load and Scale Trajectory Data
        merged_df = load_and_combine_trc_mot_data(user_dir, os.path.basename(file_path))
        merged_df = scale_coordinates(merged_df)

        # Save processed data as CSV
        data_filename = f"{os.path.splitext(file.filename)[0]}_data.csv"
        data_path = os.path.join(user_dir, data_filename)
        merged_df.to_csv(data_path, index=False)

        # Step 4: Overlay Trajectory
        joint_name = "Hip"
        output_path = os.path.join(user_dir, f"{os.path.splitext(file.filename)[0]}_processed.mp4")
        overlay_joint_trajectory(
            video_path=file_path,
            df=merged_df,
            joint_name=joint_name,
            output_path=output_path,
            frame_rate=30,
            visualize_velocity="color"
        )

        # Step 5: Clean Up Temporary Files
        os.remove(file_path)
        cleanup_static_folder()

        # Step 6: Serve the Processed Video and Data
        return jsonify({
            'status': 'Video processed successfully!',
            'video_path': f'/{output_path}',
            'data_path': f'/{data_path}'
        })
    except Exception as e:
        print(f"Error processing video: {e}")
        return jsonify({'status': 'Error processing video', 'error': str(e)})





if __name__ == '__main__':
    app.run(debug=True)
