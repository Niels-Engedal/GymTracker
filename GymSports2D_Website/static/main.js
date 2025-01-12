const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const fileInput = document.getElementById('file-input');

// Access the camera
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => console.error('Error accessing camera:', err));

// Handle file upload
document.getElementById('upload-form').onsubmit = async (e) => {
    e.preventDefault();
    const file = fileInput.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('image', file);

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();
        alert(result.status);
    }
};

const uploadForm = document.getElementById('upload-form');
const videoInput = document.getElementById('video-input');
const uploadedVideo = document.getElementById('uploaded-video');
const progressIndicator = document.getElementById('progress');

uploadForm.onsubmit = async (e) => {
    e.preventDefault();
    const file = videoInput.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('video', file);

        // Show progress indicator
        progressIndicator.style.display = 'block';

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });

            const result = await response.json();

            if (result.path) {
                uploadedVideo.src = result.path; // Display the processed video
                uploadedVideo.style.display = 'block';
            } else {
                alert('Error processing video: ' + result.error);
            }
        } catch (error) {
            alert('An error occurred: ' + error.message);
        } finally {
            progressIndicator.style.display = 'none'; // Hide progress indicator
        }
    } else {
        alert('Please select a video file to upload.');
    }
};



