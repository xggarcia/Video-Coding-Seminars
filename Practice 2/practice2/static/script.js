// Global state
let selectedFile = null;
const API_BASE = '';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const resultsSection = document.getElementById('resultsSection');
const resultsContent = document.getElementById('resultsContent');
const progressOverlay = document.getElementById('progressOverlay');
const progressText = document.getElementById('progressText');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    hideResults();
});

function setupEventListeners() {
    // Upload area
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
    
    // Tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });
    
    // Codec buttons
    document.querySelectorAll('.codec-btn').forEach(btn => {
        btn.addEventListener('click', () => convertCodec(btn.dataset.codec));
    });
    
    // Encoding ladder
    document.getElementById('createLadderBtn').addEventListener('click', createEncodingLadder);
    
    // Analysis buttons
    document.getElementById('metadataBtn').addEventListener('click', getMetadata);
    document.getElementById('motionVectorsBtn').addEventListener('click', visualizeMotionVectors);
    document.getElementById('yuvHistogramBtn').addEventListener('click', createYuvHistogram);
    document.getElementById('trackCountBtn').addEventListener('click', countTracks);
    
    // Processing buttons
    document.getElementById('resizeBtn').addEventListener('click', resizeVideo);
    document.getElementById('chromaBtn').addEventListener('click', applyChroma);
    document.getElementById('bbbBtn').addEventListener('click', createBBB);
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('video/')) {
        showError('Please select a video file');
        return;
    }
    
    selectedFile = file;
    fileInfo.innerHTML = `
        <strong>✓ File selected:</strong> ${file.name} 
        <span style="color: #6b7280;">(${(file.size / (1024 * 1024)).toFixed(2)} MB)</span>
    `;
    fileInfo.classList.remove('hidden');
}

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `${tabName}-tab`);
    });
}

function showProgress(message) {
    progressText.textContent = message;
    progressOverlay.classList.remove('hidden');
}

function hideProgress() {
    progressOverlay.classList.add('hidden');
}

function showResults() {
    resultsSection.style.display = 'block';
}

function hideResults() {
    resultsSection.style.display = 'none';
}

function showError(message) {
    showResults();
    resultsContent.innerHTML = `
        <div class="result-item result-error">
            <strong>❌ Error:</strong> ${message}
        </div>
    `;
}

function showSuccess(message, downloadUrl = null) {
    showResults();
    let html = `
        <div class="result-item result-success">
            <strong>✓ Success:</strong> ${message}
    `;
    
    if (downloadUrl) {
        html += `<br><a href="${downloadUrl}" class="download-btn" download>📥 Download Result</a>`;
    }
    
    html += `</div>`;
    resultsContent.innerHTML = html;
}

// API Functions

async function convertCodec(codec) {
    if (!selectedFile) {
        showError('Please select a video file first');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    showProgress(`Converting to ${codec.toUpperCase()}...`);
    
    try {
        const response = await fetch(`${API_BASE}/convert_codec?codec=${codec}`, {
            method: 'POST',
            body: formData
        });
        
        hideProgress();
        
        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const filename = response.headers.get('content-disposition')?.split('filename=')[1] || `video_${codec}.${getExtension(codec)}`;
            
            showSuccess(`Video converted to ${codec.toUpperCase()}!`, url);
        } else {
            const error = await response.json();
            showError(error.detail || 'Conversion failed');
        }
    } catch (error) {
        hideProgress();
        showError(`Network error: ${error.message}`);
    }
}

async function createEncodingLadder() {
    if (!selectedFile) {
        showError('Please select a video file first');
        return;
    }
    
    const codec = document.getElementById('ladderCodec').value;
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    showProgress('Creating encoding ladder... This may take a while...');
    
    try {
        const response = await fetch(`${API_BASE}/create_encoding_ladder?codec=${codec}`, {
            method: 'POST',
            body: formData
        });
        
        hideProgress();
        
        if (response.ok) {
            const data = await response.json();
            displayLadderResults(data);
        } else {
            const error = await response.json();
            showError(error.detail || 'Encoding ladder creation failed');
        }
    } catch (error) {
        hideProgress();
        showError(`Network error: ${error.message}`);
    }
}

function displayLadderResults(data) {
    showResults();
    let html = `
        <div class="result-item result-success">
            <strong>✓ ${data.message}</strong>
        </div>
        <div class="ladder-results">
    `;
    
    data.variants.forEach(variant => {
        html += `
            <div class="ladder-variant">
                <h4>${variant.resolution} - ${variant.width}x${variant.height}</h4>
                <p>Bitrate: ${variant.bitrate} | Codec: ${variant.codec}</p>
                <p><strong>File:</strong> ${variant.file_name}</p>
            </div>
        `;
    });
    
    html += '</div>';
    resultsContent.innerHTML = html;
}

async function getMetadata() {
    if (!selectedFile) {
        showError('Please select a video file first');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    showProgress('Analyzing video metadata...');
    
    try {
        const response = await fetch(`${API_BASE}/relevant_information`, {
            method: 'POST',
            body: formData
        });
        
        hideProgress();
        
        if (response.ok) {
            const text = await response.text();
            showResults();
            resultsContent.innerHTML = `
                <div class="metadata-display">
                    <h4>📊 Video Information</h4>
                    <pre>${text}</pre>
                </div>
            `;
        } else {
            showError('Failed to retrieve metadata');
        }
    } catch (error) {
        hideProgress();
        showError(`Network error: ${error.message}`);
    }
}

async function visualizeMotionVectors() {
    if (!selectedFile) {
        showError('Please select a video file first');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    showProgress('Visualizing motion vectors...');
    
    try {
        const response = await fetch(`${API_BASE}/visualize_motion_vectors`, {
            method: 'POST',
            body: formData
        });
        
        hideProgress();
        
        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            showSuccess('Motion vectors visualization created!', url);
        } else {
            const error = await response.json();
            showError(error.detail || 'Motion vector visualization failed');
        }
    } catch (error) {
        hideProgress();
        showError(`Network error: ${error.message}`);
    }
}

async function createYuvHistogram() {
    if (!selectedFile) {
        showError('Please select a video file first');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    showProgress('Creating YUV histogram...');
    
    try {
        const response = await fetch(`${API_BASE}/yuv_histogram`, {
            method: 'POST',
            body: formData
        });
        
        hideProgress();
        
        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            showSuccess('YUV histogram created!', url);
        } else {
            const error = await response.json();
            showError(error.detail || 'YUV histogram creation failed');
        }
    } catch (error) {
        hideProgress();
        showError(`Network error: ${error.message}`);
    }
}

async function countTracks() {
    if (!selectedFile) {
        showError('Please select a video file first');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    showProgress('Counting tracks...');
    
    try {
        const response = await fetch(`${API_BASE}/count-tracks`, {
            method: 'POST',
            body: formData
        });
        
        hideProgress();
        
        if (response.ok) {
            const data = await response.json();
            displayTrackInfo(data);
        } else {
            showError('Failed to count tracks');
        }
    } catch (error) {
        hideProgress();
        showError(`Network error: ${error.message}`);
    }
}

function displayTrackInfo(data) {
    showResults();
    let html = `
        <div class="result-item result-success">
            <strong>${data.message}</strong>
            <p style="margin-top: 10px;">File: ${data.filename}</p>
        </div>
    `;
    
    data.streams.forEach((stream, idx) => {
        html += `
            <div class="result-item">
                <h4>Track ${idx + 1}: ${stream.codec_type.toUpperCase()}</h4>
                <p><strong>Codec:</strong> ${stream.codec_name}</p>
                ${stream.width ? `<p><strong>Resolution:</strong> ${stream.width}x${stream.height}</p>` : ''}
                ${stream.channels ? `<p><strong>Channels:</strong> ${stream.channels}</p>` : ''}
            </div>
        `;
    });
    
    resultsContent.innerHTML = html;
}

async function resizeVideo() {
    if (!selectedFile) {
        showError('Please select a video file first');
        return;
    }
    
    const width = document.getElementById('resizeWidth').value;
    const height = document.getElementById('resizeHeight').value;
    
    if (!width || !height) {
        showError('Please enter both width and height');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    showProgress(`Resizing to ${width}x${height}...`);
    
    try {
        const response = await fetch(`${API_BASE}/resize?width=${width}&height=${height}&isVideo=true`, {
            method: 'POST',
            body: formData
        });
        
        hideProgress();
        
        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            showSuccess(`Video resized to ${width}x${height}!`, url);
        } else {
            const error = await response.json();
            showError(error.detail || 'Resize failed');
        }
    } catch (error) {
        hideProgress();
        showError(`Network error: ${error.message}`);
    }
}

async function applyChroma() {
    if (!selectedFile) {
        showError('Please select a video file first');
        return;
    }
    
    const subsampling = document.getElementById('chromaSelect').value;
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    showProgress(`Applying ${subsampling} chroma subsampling...`);
    
    try {
        const response = await fetch(`${API_BASE}/set-chroma?subsampling=${encodeURIComponent(subsampling)}`, {
            method: 'POST',
            body: formData
        });
        
        hideProgress();
        
        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            showSuccess(`Chroma subsampling ${subsampling} applied!`, url);
        } else {
            const error = await response.json();
            showError(error.detail || 'Chroma conversion failed');
        }
    } catch (error) {
        hideProgress();
        showError(`Network error: ${error.message}`);
    }
}

async function createBBB() {
    if (!selectedFile) {
        showError('Please select a video file first');
        return;
    }
    
    const duration = document.getElementById('bbbDuration').value || 20;
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    showProgress('Creating BBB container with multi-track audio...');
    
    try {
        const response = await fetch(`${API_BASE}/create_bbb_container?duration=${duration}`, {
            method: 'POST',
            body: formData
        });
        
        hideProgress();
        
        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            showSuccess(`BBB container created with AAC, MP3, and AC3 audio tracks!`, url);
        } else {
            const error = await response.json();
            showError(error.detail || 'BBB creation failed');
        }
    } catch (error) {
        hideProgress();
        showError(`Network error: ${error.message}`);
    }
}

function getExtension(codec) {
    const extensions = {
        'vp8': 'webm',
        'vp9': 'webm',
        'h265': 'mp4',
        'av1': 'mkv'
    };
    return extensions[codec] || 'mp4';
}
