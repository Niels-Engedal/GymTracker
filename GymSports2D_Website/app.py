from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    # Convert uploaded image to numpy array for OpenCV processing
    np_img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Display the image using OpenCV
    cv2.imshow('Uploaded Image', img)
    cv2.waitKey(1000)  # Display for 1 second
    cv2.destroyAllWindows()

    return jsonify({'status': 'Image displayed successfully!'})

if __name__ == '__main__':
    app.run(debug=True)
