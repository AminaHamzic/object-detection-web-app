$(function () {
    // init feather icons
    feather.replace();
});

document.getElementById('upload_button').addEventListener('click', async () => {
    let fileInput = document.getElementById('file_upload');
    let formData = new FormData();
    formData.append('file_upload', fileInput.files[0]);

    try {
        let response = await fetch('http://127.0.0.1:8000/uploadfile/', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            let data = await response.json();
            console.log('Upload successful', data);
        } else {
            console.error('Upload failed');
        }
    } catch (error) {
        console.error('Error:', error);
    }
});
