from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import subprocess
import os
import time


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['video']
    file_path = f"temp_{file.filename}"
    file.save(file_path)

    # Check if the file is a .MOV
    if file.filename.endswith('.mov') or file.filename.endswith('.MOV'):
        converted_path = f"static/converted_{os.path.splitext(file.filename)[0]}.mp4"
        with open('ffmpeg.log', 'w') as logfile:
            subprocess.run(['ffmpeg', '-y', '-i', file_path, '-c:v', 'libx264', '-c:a', 'aac', converted_path],
                        stdout=logfile, stderr=subprocess.STDOUT)


        # Clean up the original .MOV file
        os.remove(file_path)
        # Clean up old files
        cleanup_static_folder()
        
        # Return the path to the converted video
        return jsonify({'status': 'Video converted successfully!', 'path': f'/{converted_path}'})

    return jsonify({'status': 'Video processed successfully!', 'path': file_path})


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
