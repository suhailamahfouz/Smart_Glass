/**
 * Smart Glass - Main JavaScript Controller
 * =========================================
 * Professional AI Vision Assistant
 */

// ==========================================
// Application State Management
// ==========================================
const appState = {
    cameraActive: false,
    videoStream: null,
    backendUrl: 'http://localhost:5000',
    voiceLang: 'en',
    autoSpeak: true,
    backendOnline: false
};

// ==========================================
// DOM Element References
// ==========================================
const DOM = {
    video: document.getElementById('video'),
    canvas: document.getElementById('canvas'),
    cameraSelect: document.getElementById('cameraSelect'),
    btnStartCamera: document.getElementById('btnStartCamera'),
    btnStopCamera: document.getElementById('btnStopCamera'),
    placeholder: document.getElementById('placeholder'),
    overlay: document.getElementById('overlay'),
    overlayText: document.getElementById('overlayText'),
    loading: document.getElementById('loading'),
    loadingText: document.getElementById('loadingText'),
    resultContent: document.getElementById('resultContent'),
    audioPlayer: document.getElementById('audioPlayer'),
    cameraStatusBadge: document.getElementById('cameraStatusBadge'),
    statusIcon: document.getElementById('statusIcon'),
    statusText: document.getElementById('statusText'),
    backendStatus: document.getElementById('backendStatus'),
    backendUrlInput: document.getElementById('backendUrlInput'),
    voiceLang: document.getElementById('voiceLang'),
    autoSpeak: document.getElementById('autoSpeak'),
    btnTestBackend: document.getElementById('btnTestBackend'),
    toastContainer: document.getElementById('toastContainer')
};

// Feature Buttons
const featureButtons = {
    time: document.getElementById('btnTime'),
    objects: document.getElementById('btnObjects'),
    faces: document.getElementById('btnFaces'),
    color: document.getElementById('btnColor'),
    text: document.getElementById('btnText'),
    analysis: document.getElementById('btnAnalysis')
};

// ==========================================
// Utility Functions
// ==========================================

/**
 * Show Bootstrap Toast Notification
 */
function showToast(message, type = 'info') {
    const icons = {
        success: 'bi-check-circle-fill',
        error: 'bi-x-circle-fill',
        info: 'bi-info-circle-fill',
        warning: 'bi-exclamation-triangle-fill'
    };

    const colors = {
        success: 'text-success',
        error: 'text-danger',
        info: 'text-info',
        warning: 'text-warning'
    };

    const toastId = 'toast-' + Date.now();
    const toastHTML = `
        <div class="toast toast-${type}" role="alert" id="${toastId}" data-bs-autohide="true" data-bs-delay="4000">
            <div class="toast-body d-flex align-items-center">
                <i class="bi ${icons[type]} ${colors[type]} fs-4 me-3"></i>
                <div class="flex-grow-1">${message}</div>
                <button type="button" class="btn-close" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;

    DOM.toastContainer.insertAdjacentHTML('beforeend', toastHTML);
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement);
    toast.show();

    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

/**
 * Show/Hide Loading Indicator
 */
function setLoading(show, text = 'Processing...') {
    DOM.loading.style.display = show ? 'block' : 'none';
    if (show) DOM.loadingText.textContent = text;
}

/**
 * Update Video Overlay Text
 */
function updateOverlay(text) {
    DOM.overlayText.textContent = text;
}

/**
 * Display Results
 */
function showResult(content) {
    DOM.resultContent.innerHTML = content;
}

/**
 * Text-to-Speech
 */
function speak(text) {
    if (!appState.autoSpeak) return;
    
    if ('speechSynthesis' in window) {
        speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = appState.voiceLang === 'ar' ? 'ar-SA' : 'en-US';
        utterance.rate = 0.9;
        utterance.pitch = 1;
        speechSynthesis.speak(utterance);
    }
}

/**
 * Capture Frame from Video
 */
function captureFrame() {
    if (!appState.cameraActive) {
        showToast('Please start the camera first', 'warning');
        return null;
    }
    
    const video = DOM.video;
    const canvas = DOM.canvas;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    return canvas.toDataURL('image/jpeg', 0.8);
}

/**
 * Enable/Disable AI Feature Buttons
 */
function toggleFeatureButtons(enable) {
    Object.values(featureButtons).forEach(btn => {
        if (enable) {
            btn.classList.add('enabled');
        } else {
            btn.classList.remove('enabled');
        }
    });
}

/**
 * Update Camera Status Badge
 */
function updateCameraStatus(active) {
    if (active) {
        DOM.cameraStatusBadge.classList.remove('inactive');
        DOM.cameraStatusBadge.classList.add('active');
        DOM.statusText.textContent = 'Active';
    } else {
        DOM.cameraStatusBadge.classList.remove('active');
        DOM.cameraStatusBadge.classList.add('inactive');
        DOM.statusText.textContent = 'Inactive';
    }
}

// ==========================================
// Camera Management Functions
// ==========================================

/**
 * Load Available Cameras
 */
async function loadCameras() {
    try {
        // Request permission
        const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
        tempStream.getTracks().forEach(track => track.stop());
        
        // Enumerate devices
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        
        DOM.cameraSelect.innerHTML = '<option value="">Choose a camera...</option>';
        
        videoDevices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || `Camera ${index + 1}`;
            DOM.cameraSelect.appendChild(option);
        });
        
        if (videoDevices.length > 0) {
            DOM.cameraSelect.value = videoDevices[0].deviceId;
        }
        
        showToast('Cameras loaded successfully', 'success');
    } catch (err) {
        showToast('Error accessing cameras: ' + err.message, 'error');
        console.error('Camera error:', err);
    }
}

/**
 * Start Camera Stream
 */
async function startCamera() {
    const deviceId = DOM.cameraSelect.value;
    
    if (!deviceId) {
        showToast('Please select a camera first', 'warning');
        return;
    }
    
    try {
        setLoading(true, 'Starting camera...');
        
        // Stop existing stream
        if (appState.videoStream) {
            appState.videoStream.getTracks().forEach(track => track.stop());
        }
        
        // Request new stream
        const constraints = {
            video: {
                deviceId: { exact: deviceId },
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        };
        
        appState.videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        DOM.video.srcObject = appState.videoStream;
        
        // Update UI
        DOM.placeholder.style.display = 'none';
        DOM.video.classList.add('show');
        DOM.overlay.classList.add('show');
        DOM.btnStartCamera.style.display = 'none';
        DOM.btnStopCamera.style.display = 'inline-block';
        
        // Update state
        appState.cameraActive = true;
        toggleFeatureButtons(true);
        updateCameraStatus(true);
        
        updateOverlay('Camera Ready ✓');
        showResult('<div class="text-success"><i class="bi bi-check-circle me-2"></i>Camera is active! You can now use AI features.</div>');
        
        showToast('Camera started successfully', 'success');
        
    } catch (err) {
        showToast('Error starting camera: ' + err.message, 'error');
        console.error('Start camera error:', err);
    } finally {
        setLoading(false);
    }
}

/**
 * Stop Camera Stream
 */
function stopCamera() {
    if (appState.videoStream) {
        appState.videoStream.getTracks().forEach(track => track.stop());
        appState.videoStream = null;
    }
    
    // Update UI
    DOM.video.classList.remove('show');
    DOM.overlay.classList.remove('show');
    DOM.placeholder.style.display = 'block';
    DOM.btnStartCamera.style.display = 'inline-block';
    DOM.btnStopCamera.style.display = 'none';
    
    // Update state
    appState.cameraActive = false;
    toggleFeatureButtons(false);
    updateCameraStatus(false);
    
    showResult('Camera stopped. Click "Start Camera" to resume.');
    showToast('Camera stopped', 'info');
}

// ==========================================
// Backend API Functions
// ==========================================

/**
 * Check Backend Server Status
 */
async function checkBackendStatus() {
    try {
        const response = await fetch(`${appState.backendUrl}/api/health`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        appState.backendOnline = true;
        DOM.backendStatus.innerHTML = `
            <span class="text-success">
                <i class="bi bi-check-circle-fill"></i> Connected - ${data.known_faces || 0} known faces
            </span>
        `;
        
        return true;
    } catch (err) {
        appState.backendOnline = false;
        DOM.backendStatus.innerHTML = `
            <span class="text-danger">
                <i class="bi bi-x-circle-fill"></i> Disconnected - Ensure backend is running
            </span>
        `;
        
        return false;
    }
}

/**
 * Send Request to Backend API
 */
async function sendToBackend(endpoint, imageData) {
    try {
        const response = await fetch(`${appState.backendUrl}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        return await response.json();
        
    } catch (err) {
        throw new Error(`Backend connection failed: ${err.message}`);
    }
}

// ==========================================
// AI Feature Functions
// ==========================================

/**
 * Tell Current Time
 */
function tellTime() {
    if (!appState.cameraActive) {
        showToast('Camera must be active', 'warning');
        return;
    }
    
    updateOverlay('🕐 Time');
    
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit',
        hour12: true 
    });
    const dateStr = now.toLocaleDateString('en-US', { 
        weekday: 'long', 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric' 
    });
    
    const result = `The time is ${timeStr}. Today is ${dateStr}`;
    
    showResult(`
        <div class="result-item">
            <strong><i class="bi bi-clock me-2"></i>Time:</strong> ${timeStr}
        </div>
        <div class="result-item">
            <strong><i class="bi bi-calendar me-2"></i>Date:</strong> ${dateStr}
        </div>
    `);
    
    speak(result);
}

/**
 * Detect Objects in Frame
 */
async function detectObjects() {
    if (!appState.cameraActive) {
        showToast('Camera must be active', 'warning');
        return;
    }
    
    if (!appState.backendOnline) {
        showToast('Backend is not connected', 'error');
        return;
    }
    
    try {
        updateOverlay('📦 Detecting Objects');
        setLoading(true, 'Detecting objects...');
        
        const imageData = captureFrame();
        const result = await sendToBackend('/api/detect-objects', imageData);
        
        if (result.success) {
            let html = `<div class="mb-3"><strong>Found ${result.count} object(s):</strong></div>`;
            
            result.objects.forEach((obj, index) => {
                html += `
                    <div class="result-item">
                        <i class="bi bi-box me-2"></i>${index + 1}. ${obj.class_name} (${obj.confidence}%)
                    </div>
                `;
            });
            
            showResult(html);
            
            const objectNames = result.objects.map(obj => obj.class_name).join(', ');
            speak(`There are ${result.count} objects: ${objectNames}`);
            
            showToast('Objects detected successfully', 'success');
        } else {
            showToast('Failed to detect objects', 'error');
        }
        
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(false);
    }
}

/**
 * Recognize Faces in Frame
 */
async function recognizeFaces() {
    if (!appState.cameraActive) {
        showToast('Camera must be active', 'warning');
        return;
    }
    
    if (!appState.backendOnline) {
        showToast('Backend is not connected', 'error');
        return;
    }
    
    try {
        updateOverlay('👤 Recognizing Faces');
        setLoading(true, 'Recognizing faces...');
        
        const imageData = captureFrame();
        const result = await sendToBackend('/api/recognize-faces', imageData);
        
        if (result.success) {
            let html = `<div class="mb-3"><strong>Found ${result.count} face(s):</strong></div>`;
            
            result.faces.forEach((face, index) => {
                const confidence = face.confidence ? ` (${face.confidence}%)` : '';
                html += `
                    <div class="result-item">
                        <i class="bi bi-person me-2"></i>${index + 1}. ${face.name}${confidence}
                    </div>
                `;
            });
            
            showResult(html);
            
            const faceNames = result.faces.map(f => f.name).join(', ');
            speak(`There are ${result.count} faces: ${faceNames}`);
            
            showToast('Faces recognized successfully', 'success');
        } else {
            showToast('Failed to recognize faces', 'error');
        }
        
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(false);
    }
}

/**
 * Detect Dominant Color
 */
async function detectColor() {
    if (!appState.cameraActive) {
        showToast('Camera must be active', 'warning');
        return;
    }
    
    if (!appState.backendOnline) {
        showToast('Backend is not connected', 'error');
        return;
    }
    
    try {
        updateOverlay('🎨 Detecting Color');
        setLoading(true, 'Analyzing color...');
        
        const imageData = captureFrame();
        const result = await sendToBackend('/api/detect-color', imageData);
        
        if (result.success) {
            const rgb = result.rgb;
            const colorStyle = `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`;
            
            const html = `
                <div class="result-item">
                    <strong><i class="bi bi-palette me-2"></i>Detected Color:</strong> ${result.color_name}
                </div>
                <div class="result-item">
                    <strong>RGB Values:</strong> ${rgb.r}, ${rgb.g}, ${rgb.b}
                </div>
                <div class="text-center mt-3">
                    <div style="width: 120px; height: 120px; background: ${colorStyle}; 
                         border-radius: 15px; margin: 0 auto; border: 3px solid #dee2e6; 
                         box-shadow: 0 5px 20px rgba(0,0,0,0.15);">
                    </div>
                </div>
            `;
            
            showResult(html);
            speak(`The dominant color is ${result.color_name}`);
            
            showToast('Color detected successfully', 'success');
        } else {
            showToast('Failed to detect color', 'error');
        }
        
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(false);
    }
}

/**
 * Read Text from Image (OCR)
 */
async function readText() {
    if (!appState.cameraActive) {
        showToast('Camera must be active', 'warning');
        return;
    }
    
    if (!appState.backendOnline) {
        showToast('Backend is not connected', 'error');
        return;
    }
    
    try {
        updateOverlay('📝 Reading Text');
        setLoading(true, 'Reading text...');
        
        const imageData = captureFrame();
        const result = await sendToBackend('/api/read-text', imageData);
        
        if (result.success) {
            if (result.has_text) {
                const html = `
                    <div class="result-item">
                        <strong><i class="bi bi-file-text me-2"></i>Detected Text:</strong><br>
                        <div class="mt-2 p-3 bg-light rounded">
                            ${result.text.replace(/\n/g, '<br>')}
                        </div>
                    </div>
                `;
                
                showResult(html);
                speak(result.text);
                
                showToast('Text read successfully', 'success');
            } else {
                showResult('<div class="text-muted"><i class="bi bi-info-circle me-2"></i>No text found in the image</div>');
                speak('No text found');
                showToast('No text found', 'info');
            }
        } else {
            showToast('Failed to read text', 'error');
        }
        
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(false);
    }
}

/**
 * Perform Full Image Analysis
 */
async function fullAnalysis() {
    if (!appState.cameraActive) {
        showToast('Camera must be active', 'warning');
        return;
    }
    
    if (!appState.backendOnline) {
        showToast('Backend is not connected', 'error');
        return;
    }
    
    try {
        updateOverlay('🔍 Full Analysis');
        setLoading(true, 'Performing full analysis...');
        
        const imageData = captureFrame();
        const result = await sendToBackend('/api/full-analysis', imageData);
        
        if (result.success) {
            let html = `<div class="mb-3"><strong>${result.description}</strong></div>`;
            
            if (result.objects && result.objects.length > 0) {
                html += '<div class="mt-3"><strong><i class="bi bi-box-seam me-2"></i>Objects:</strong></div>';
                result.objects.forEach(obj => {
                    html += `<div class="result-item">${obj.name}</div>`;
                });
            }
            
            if (result.faces && result.faces.length > 0) {
                html += '<div class="mt-3"><strong><i class="bi bi-people me-2"></i>Faces:</strong></div>';
                result.faces.forEach(face => {
                    html += `<div class="result-item">${face.name}</div>`;
                });
            }
            
            if (result.color) {
                html += `<div class="result-item mt-3"><strong><i class="bi bi-palette me-2"></i>Dominant Color:</strong> ${result.color}</div>`;
            }
            
            showResult(html);
            speak(result.description);
            
            showToast('Analysis completed successfully', 'success');
        } else {
            showToast('Analysis failed', 'error');
        }
        
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading(false);
    }
}

// ==========================================
// Event Listeners
// ==========================================

// Camera Controls
DOM.btnStartCamera.addEventListener('click', startCamera);
DOM.btnStopCamera.addEventListener('click', stopCamera);

// AI Feature Buttons
featureButtons.time.addEventListener('click', tellTime);
featureButtons.objects.addEventListener('click', detectObjects);
featureButtons.faces.addEventListener('click', recognizeFaces);
featureButtons.color.addEventListener('click', detectColor);
featureButtons.text.addEventListener('click', readText);
featureButtons.analysis.addEventListener('click', fullAnalysis);

// Settings
DOM.backendUrlInput.addEventListener('change', (e) => {
    appState.backendUrl = e.target.value;
    checkBackendStatus();
});

DOM.voiceLang.addEventListener('change', (e) => {
    appState.voiceLang = e.target.value;
});

DOM.autoSpeak.addEventListener('change', (e) => {
    appState.autoSpeak = e.target.checked;
});

DOM.btnTestBackend.addEventListener('click', async () => {
    setLoading(true, 'Testing connection...');
    const online = await checkBackendStatus();
    setLoading(false);
    
    if (online) {
        showToast('Backend connection successful', 'success');
    } else {
        showToast('Backend connection failed', 'error');
    }
});

// ==========================================
// Application Initialization
// ==========================================
window.addEventListener('load', async () => {
    console.log('Smart Glass Application Initialized');
    
    // Load available cameras
    await loadCameras();
    
    // Check backend status
    await checkBackendStatus();
    
    // Set initial UI state
    showResult('<div class="text-muted"><i class="bi bi-info-circle me-2"></i>Waiting for camera activation...</div>');
    updateCameraStatus(false);
    
    // Welcome message
    showToast('Welcome! Start the camera to begin.', 'info');
});