$(function () {
    // init feather icons
    feather.replace();
});

document.addEventListener('DOMContentLoaded', (event) => {
    const fileInput = document.getElementById('file-input');
    const modelConfidenceInput = document.getElementById('model-confidence');

    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        const reader = new FileReader();

        reader.onload = (e) => {
            const image = new Image();
            image.src = e.target.result;

            image.onload = () => {
                // Display the image
                const imageContainer = document.getElementById('image-container');
                imageContainer.innerHTML = '';
                imageContainer.appendChild(image);

                // Display the image dimensions
                const imageDimensions = document.getElementById('image-dimensions');
                imageDimensions.textContent = `Image dimensions: ${image.width} x ${image.height}`;

                // Display the file size
                const fileSize = document.getElementById('file-size');
                fileSize.textContent = `File size: ${file.size} bytes`;
            };
        };

        reader.readAsDataURL(file);
    });

    modelConfidenceInput.addEventListener('input', (event) => {
        // Update the model confidence value, if you need to display it
        console.log('Model confidence:', event.target.value);
    });
});

