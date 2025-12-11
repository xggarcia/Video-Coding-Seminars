// Professional VideoForge Pro JavaScript
let uploadedFile = null;
let currentTab = 'codec';

// Theme Toggle
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    html.setAttribute('data-theme', currentTheme === 'dark' ? 'light' : 'dark');
    localStorage.setItem('theme', currentTheme === 'dark' ? 'light' : 'dark');
}

// Initialize theme
document.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    
    // Setup file input
    setupFileUpload();
    // Setup tabs
    setupTabs();
});

// File Upload
function setupFileUpload() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const selectBtn = document.getElementById('selectFileBtn');

    // Click button to upload
    if (selectBtn) {
        selectBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            fileInput.click();
        });
    }

    // Don't add click to zone - button is enough

    // File selection
    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--primary)';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = 'var(--border)';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border)';
        handleFile(e.dataTransfer.files[0]);
    });
}

function handleFile(file) {
    if (!file) return;
    uploadedFile = file;
    const fileInfo = document.getElementById('fileInfo');
    
    // Determine icon based on file type
    let icon = '📄';
    if (file.type.startsWith('video/')) {
        icon = '🎬';
    } else if (file.type.startsWith('image/')) {
        icon = '🖼️';
    }
    
    fileInfo.innerHTML = `
        <strong>${icon} ${file.name}</strong> - ${(file.size / 1024 / 1024).toFixed(2)} MB
    `;
    fileInfo.classList.add('active');
}

// Tabs
function setupTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });
}

function switchTab(tabName) {
    currentTab = tabName;
    
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });
    
    // Update panels
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.toggle('active', panel.id === `${tabName}-panel`);
    });
}

// Show/Hide Loading
function showLoading(status = 'Processing...') {
    const loading = document.getElementById('loading');
    const statusText = document.getElementById('statusText');
    statusText.textContent = status;
    loading.classList.add('active');
}

function hideLoading() {
    document.getElementById('loading').classList.remove('active');
}

// Show Results
function showResults(content, isHTML = false) {
    hideLoading();
    const results = document.getElementById('results');
    const resultsContent = document.getElementById('resultsContent');
    
    if (isHTML) {
        resultsContent.innerHTML = content;
    } else {
        resultsContent.innerHTML = `<pre>${content}</pre>`;
    }
    
    results.classList.add('active');
    results.scrollIntoView({ behavior: 'smooth' });
}

function closeResults() {
    document.getElementById('results').classList.remove('active');
}

// API Calls
async function makeAPICall(endpoint, formData, status) {
    if (!uploadedFile) {
        alert('Please upload a file first!');
        return;
    }
    
    showLoading(status);
    
    try {
        if (!formData.has('file')) {
            formData.append('file', uploadedFile);
        }
        
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }
        
        const contentType = response.headers.get('content-type');
        
        if (contentType && contentType.includes('application/json')) {
            const data = await response.json();
            showResults(JSON.stringify(data, null, 2));
        } else {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const filename = response.headers.get('content-disposition')?.match(/filename="(.+)"/)?.[1] || 'output.mp4';
            
            showResults(`
                <div style="text-align: center;">
                    <h3>✅ Processing Complete!</h3>
                    <p style="margin: 1rem 0;">File: ${filename}</p>
                    <video controls style="max-width: 100%; max-height: 400px; border-radius: 8px; margin: 1rem 0;" src="${url}"></video>
                    <br>
                    <a href="${url}" download="${filename}" style="display: inline-block; background: var(--primary); color: white; padding: 0.75rem 2rem; border-radius: 8px; text-decoration: none; font-weight: 600;">
                        ⬇️ Download File
                    </a>
                </div>
            `, true);
        }
    } catch (error) {
        hideLoading();
        alert(`Error: ${error.message}`);
    }
}

// Codec Conversion
async function convertCodec(codec) {
    const formData = new FormData();
    await makeAPICall(`/convert_codec?codec=${codec}`, formData, `Converting to ${codec}...`);
}

// Encoding Ladder
async function createLadder(codec) {
    const formData = new FormData();
    await makeAPICall(`/create_encoding_ladder?codec=${codec}`, formData, `Creating ${codec} encoding ladder...`);
}

// Video Info
async function getVideoInfo() {
    const formData = new FormData();
    await makeAPICall('/relevant_information', formData, 'Analyzing video...');
}

// YUV Histogram
async function generateHistogram() {
    const formData = new FormData();
    await makeAPICall('/yuv_histogram', formData, 'Generating YUV histogram...');
}

// Motion Vectors
async function visualizeMotionVectors() {
    const formData = new FormData();
    await makeAPICall('/visualize_motion_vectors', formData, 'Visualizing motion vectors...');
}

// Count Tracks
async function countTracks() {
    const formData = new FormData();
    await makeAPICall('/count_tracks', formData, 'Counting tracks...');
}

// DCT Transform
async function applyDCT() {
    const formData = new FormData();
    await makeAPICall('/process-dct', formData, 'Applying DCT transform...');
}

// DWT Transform
async function applyDWT() {
    const formData = new FormData();
    await makeAPICall('/process-dwt', formData, 'Applying DWT transform...');
}

// Serpentine Read
async function applySerpentine() {
    const formData = new FormData();
    await makeAPICall('/serpentine-read', formData, 'Applying serpentine pattern...');
}

// Resize Video
async function resizeVideo() {
    const width = document.getElementById('resizeWidth').value;
    const height = document.getElementById('resizeHeight').value;
    
    if (!width || !height) {
        alert('Please enter both width and height!');
        return;
    }
    
    const isVideo = uploadedFile.type.startsWith('video/');
    const formData = new FormData();
    await makeAPICall(`/resize?width=${width}&height=${height}&isVideo=${isVideo}`, formData, 'Resizing...');
}

// Chroma Subsampling
async function setChroma(mode) {
    const formData = new FormData();
    await makeAPICall(`/set-chroma?subsampling=${mode}`, formData, `Setting chroma to ${mode}...`);
}

// RGB to YUV
async function rgbToYuv() {
    const r = document.getElementById('rgbR').value;
    const g = document.getElementById('rgbG').value;
    const b = document.getElementById('rgbB').value;
    
    if (r === '' || g === '' || b === '') {
        alert('Please enter all RGB values!');
        return;
    }
    
    showLoading('Converting RGB to YUV...');
    
    try {
        const response = await fetch(`/rgb-to-yuv?r=${r}&g=${g}&b=${b}`);
        const data = await response.json();
        showResults(`RGB(${r}, ${g}, ${b}) → YUV(${data.y}, ${data.u}, ${data.v})`);
    } catch (error) {
        hideLoading();
        alert(`Error: ${error.message}`);
    }
}

// YUV to RGB
async function yuvToRgb() {
    const y = document.getElementById('yuvY').value;
    const u = document.getElementById('yuvU').value;
    const v = document.getElementById('yuvV').value;
    
    if (y === '' || u === '' || v === '') {
        alert('Please enter all YUV values!');
        return;
    }
    
    showLoading('Converting YUV to RGB...');
    
    try {
        const response = await fetch(`/yuv-to-rgb?y=${y}&u=${u}&v=${v}`);
        const data = await response.json();
        showResults(`YUV(${y}, ${u}, ${v}) → RGB(${data.r}, ${data.g}, ${data.b})`);
    } catch (error) {
        hideLoading();
        alert(`Error: ${error.message}`);
    }
}

// BBB Container
async function createBBBContainer() {
    const formData = new FormData();
    await makeAPICall('/create_bbb_container', formData, 'Creating BBB container...');
}
