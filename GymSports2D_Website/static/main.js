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
