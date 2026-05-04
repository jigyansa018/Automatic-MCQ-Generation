document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.getElementById('file');
    const fileLabel = document.querySelector('.file-upload-label span');

    if (fileInput && fileLabel) {
        fileInput.addEventListener('change', function () {
            const files = this.files;

            if (files.length === 0) {
                // Reset to default text if no file selected
                fileLabel.textContent = '📁 Click to upload PDF or TXT files';
            } else if (files.length === 1) {
                // Show single filename
                fileLabel.textContent = '📄 ' + files[0].name;
            } else {
                // Show multiple filenames separated by commas
                const names = Array.from(files).map(f => f.name).join(', ');
                fileLabel.textContent = '📄 ' + names;
            }
        });
    }
});