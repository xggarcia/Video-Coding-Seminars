// Global variable to store the uploaded file
let uploadedFile = null;

// Show file name when selected
document.getElementById('videoFile').addEventListener('change', function(e) {
    uploadedFile = e.target.files[0];
    if (uploadedFile) {
        document.getElementById('fileName').textContent = 'Selected: ' + uploadedFile.name;
    }
});

// Convert video codec
function convertCodec(codec) {
    if (!uploadedFile) {
        alert('Please upload a video first!');
        return;
    }

    showLoading();
    
    const formData = new FormData();
    formData.append('file', uploadedFile);
    formData.append('codec', codec);

    fetch('/convert_codec?codec=' + codec, {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        hideLoading();
        downloadFile(blob, 'converted_' + codec + getExtension(codec));
        showResult('Conversion complete! File downloaded.');
    })
    .catch(error => {
        hideLoading();
        showResult('Error: ' + error);
    });
}

// Create encoding ladder
function createLadder(codec) {
    if (!uploadedFile) {
        alert('Please upload a video first!');
        return;
    }

    showLoading();
    
    const formData = new FormData();
    formData.append('file', uploadedFile);

    fetch('/create_encoding_ladder?codec=' + codec, {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        hideLoading();
        downloadFile(blob, 'encoding_ladder.zip');
        showResult('Encoding ladder created! ZIP file downloaded with 5 resolutions.');
    })
    .catch(error => {
        hideLoading();
        showResult('Error: ' + error);
    });
}

// Helper functions
function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('result').innerHTML = '';
}

function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

function showResult(message) {
    document.getElementById('result').innerHTML = message;
}

function downloadFile(blob, filename) {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
}

function getExtension(codec) {
    const extensions = {
        'vp8': '.webm',
        'vp9': '.webm',
        'h265': '.mp4',
        'av1': '.mkv'
    };
    return extensions[codec] || '.mp4';
}

// Resize video
function resizeVideo() {
    if (!uploadedFile) {
        alert('Please upload a video first!');
        return;
    }

    const width = document.getElementById('width').value;
    const height = document.getElementById('height').value;

    showLoading();
    
    const formData = new FormData();
    formData.append('file', uploadedFile);

    fetch('/resize?width=' + width + '&height=' + height + '&isVideo=true', {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        hideLoading();
        downloadFile(blob, 'resized_' + width + 'x' + height + '.mp4');
        showResult('Video resized to ' + width + 'x' + height);
    })
    .catch(error => {
        hideLoading();
        showResult('Error: ' + error);
    });
}

// Change chroma subsampling
function changeChroma(format) {
    if (!uploadedFile) {
        alert('Please upload a video first!');
        return;
    }

    showLoading();
    
    const formData = new FormData();
    formData.append('file', uploadedFile);

    fetch('/chroma?subsampling=' + encodeURIComponent(format), {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        hideLoading();
        downloadFile(blob, 'chroma_' + format.replace(/:/g, '') + '.mp4');
        showResult('Chroma subsampling changed to ' + format);
    })
    .catch(error => {
        hideLoading();
        showResult('Error: ' + error);
    });
}

// Get video info
function getVideoInfo() {
    if (!uploadedFile) {
        alert('Please upload a video first!');
        return;
    }

    showLoading();
    
    const formData = new FormData();
    formData.append('file', uploadedFile);

    fetch('/relevant_information', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(data => {
        hideLoading();
        showResult('<h3>Video Information:</h3><pre>' + data + '</pre>');
    })
    .catch(error => {
        hideLoading();
        showResult('Error: ' + error);
    });
}

// Get YUV histogram
function getHistogram() {
    if (!uploadedFile) {
        alert('Please upload a video first!');
        return;
    }

    showLoading();
    
    const formData = new FormData();
    formData.append('file', uploadedFile);

    fetch('/yuv_histogram', {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        hideLoading();
        downloadFile(blob, 'histogram.png');
        showResult('Histogram generated and downloaded!');
    })
    .catch(error => {
        hideLoading();
        showResult('Error: ' + error);
    });
}

// Get motion vectors
function getMotionVectors() {
    if (!uploadedFile) {
        alert('Please upload a video first!');
        return;
    }

    showLoading();
    
    const formData = new FormData();
    formData.append('file', uploadedFile);

    fetch('/motion_vectors', {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        hideLoading();
        downloadFile(blob, 'motion_vectors.mp4');
        showResult('Motion vectors video downloaded!');
    })
    .catch(error => {
        hideLoading();
        showResult('Error: ' + error);
    });
}

// Count tracks
function countTracks() {
    if (!uploadedFile) {
        alert('Please upload a video first!');
        return;
    }

    showLoading();
    
    const formData = new FormData();
    formData.append('file', uploadedFile);

    fetch('/count_tracks', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        showResult('Track count: ' + JSON.stringify(data, null, 2));
    })
    .catch(error => {
        hideLoading();
        showResult('Error: ' + error);
    });
}

// Apply DCT
function applyDCT() {
    if (!uploadedFile) {
        alert('Please upload an image first!');
        return;
    }

    showLoading();
    
    const formData = new FormData();
    formData.append('file', uploadedFile);

    fetch('/apply_dct', {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        hideLoading();
        downloadFile(blob, 'dct_output.jpg');
        showResult('DCT applied and downloaded!');
    })
    .catch(error => {
        hideLoading();
        showResult('Error: ' + error);
    });
}

// Apply DWT
function applyDWT() {
    if (!uploadedFile) {
        alert('Please upload an image first!');
        return;
    }

    showLoading();
    
    const formData = new FormData();
    formData.append('file', uploadedFile);

    fetch('/apply_dwt', {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        hideLoading();
        downloadFile(blob, 'dwt_output.jpg');
        showResult('DWT applied and downloaded!');
    })
    .catch(error => {
        hideLoading();
        showResult('Error: ' + error);
    });
}

// Serpentine read
function serpentineRead() {
    if (!uploadedFile) {
        alert('Please upload an image first!');
        return;
    }

    showLoading();
    
    const formData = new FormData();
    formData.append('file', uploadedFile);

    fetch('/serpentine_read', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        showResult('<h3>Serpentine Read Result:</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>');
    })
    .catch(error => {
        hideLoading();
        showResult('Error: ' + error);
    });
}

// RGB to YUV conversion
function rgbToYuv() {
    const r = document.getElementById('r').value;
    const g = document.getElementById('g').value;
    const b = document.getElementById('b').value;

    showLoading();

    fetch('/rgb_to_yuv?r=' + r + '&g=' + g + '&b=' + b)
    .then(response => response.json())
    .then(data => {
        hideLoading();
        showResult('<h3>RGB to YUV:</h3>' +
                   'RGB(' + r + ', ' + g + ', ' + b + ') = ' +
                   'YUV(' + data.Y + ', ' + data.U + ', ' + data.V + ')');
    })
    .catch(error => {
        hideLoading();
        showResult('Error: ' + error);
    });
}

// YUV to RGB conversion
function yuvToRgb() {
    const y = document.getElementById('r').value;
    const u = document.getElementById('g').value;
    const v = document.getElementById('b').value;

    showLoading();

    fetch('/yuv_to_rgb?y=' + y + '&u=' + u + '&v=' + v)
    .then(response => response.json())
    .then(data => {
        hideLoading();
        showResult('<h3>YUV to RGB:</h3>' +
                   'YUV(' + y + ', ' + u + ', ' + v + ') = ' +
                   'RGB(' + data.R + ', ' + data.G + ', ' + data.B + ')');
    })
    .catch(error => {
        hideLoading();
        showResult('Error: ' + error);
    });
}
