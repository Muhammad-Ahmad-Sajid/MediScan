const API_URL = 'http://localhost:8080';
let currentToken = localStorage.getItem('cortexray_token');
let currentPatientId = null;

// --- DOM ELEMENTS ---
const views = {
    login: document.getElementById('login-view'),
    dashboard: document.getElementById('dashboard-view')
};

const tabs = {
    'upload-tab': document.getElementById('upload-tab'),
    'patients-tab': document.getElementById('patients-tab')
};

const ui = {
    loginForm: document.getElementById('login-form'),
    loginBtnText: document.querySelector('#login-btn .btn-text'),
    loginLoader: document.querySelector('#login-btn .loader'),
    logoutBtn: document.getElementById('logout-btn'),
    navLinks: document.querySelectorAll('.nav-links li'),
    toast: document.getElementById('toast'),
    patientSelector: document.getElementById('patient-selector-container'),
    patientSelect: document.getElementById('patient-select'),
    patientsTableBody: document.getElementById('patients-table-body'),
    
    // Upload & Results
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('file-input'),
    useSampleBtn: document.getElementById('use-sample-btn'),
    resultsContainer: document.getElementById('results-container'),
    resetScanBtn: document.getElementById('reset-scan-btn'),
    
    // Result Data
    originalImg: document.getElementById('original-image'),
    heatmapImg: document.getElementById('heatmap-image'),
    confBadge: document.getElementById('confidence-badge'),
    fracStatus: document.getElementById('fracture-status'),
    valRegion: document.getElementById('val-region'),
    valConf: document.getElementById('val-confidence'),
    valRecovery: document.getElementById('val-recovery'),
    valIntervention: document.getElementById('val-intervention'),
    valTreatment: document.getElementById('val-treatment'),
    downloadReportBtn: document.getElementById('download-report-btn'),

    // Modal
    newPatientBtn: document.getElementById('new-patient-btn'),
    newPatientModal: document.getElementById('new-patient-modal'),
    cancelPatientBtn: document.getElementById('cancel-patient-btn'),
    newPatientForm: document.getElementById('new-patient-form')
};

// --- INITIALIZATION ---
function init() {
    if (currentToken) {
        showDashboard();
    } else {
        showLogin();
    }
    setupEventListeners();
}

// --- UTILS ---
function showToast(message, type='success') {
    ui.toast.textContent = message;
    ui.toast.className = `toast ${type}`;
    ui.toast.classList.remove('hidden');
    setTimeout(() => ui.toast.classList.add('hidden'), 4000);
}

function showLogin() {
    views.login.classList.add('active');
    views.login.classList.remove('hidden');
    views.dashboard.classList.add('hidden');
    views.dashboard.classList.remove('active');
}

function showDashboard() {
    views.login.classList.add('hidden');
    views.login.classList.remove('active');
    views.dashboard.classList.add('active');
    views.dashboard.classList.remove('hidden');
    loadPatients();
}

function switchTab(targetId) {
    ui.navLinks.forEach(l => l.classList.remove('active'));
    document.querySelector(`[data-target="${targetId}"]`).classList.add('active');
    
    Object.values(tabs).forEach(t => t.classList.add('hidden'));
    tabs[targetId].classList.remove('hidden');

    // Show/hide patient selector in topbar
    if(targetId === 'upload-tab') {
        ui.patientSelector.classList.remove('hidden');
    } else {
        ui.patientSelector.classList.add('hidden');
    }
}

// --- API CALLS ---
async function fetchAPI(endpoint, options = {}) {
    const headers = { ...options.headers };
    if (currentToken) {
        headers['Authorization'] = `Bearer ${currentToken}`;
    }

    try {
        const response = await fetch(`${API_URL}${endpoint}`, { ...options, headers });
        if (response.status === 401) {
            if (endpoint !== '/auth/login') {
                handleLogout();
                throw new Error('Session expired');
            } else {
                throw new Error('Incorrect email or password');
            }
        }
        
        // Handle Blob/PDF responses
        if (options.responseType === 'blob') {
            return await response.blob();
        }

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Request failed');
        return data;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// --- EVENT LISTENERS ---
function setupEventListeners() {
    // Login
    ui.loginForm.addEventListener('submit', handleLogin);
    ui.logoutBtn.addEventListener('click', handleLogout);

    // Navigation
    ui.navLinks.forEach(link => {
        link.addEventListener('click', (e) => switchTab(e.currentTarget.dataset.target));
    });

    // Patient Dropdown Change
    ui.patientSelect.addEventListener('change', (e) => {
        currentPatientId = e.target.value;
    });

    // Modal
    ui.newPatientBtn.addEventListener('click', () => ui.newPatientModal.classList.remove('hidden'));
    ui.cancelPatientBtn.addEventListener('click', () => ui.newPatientModal.classList.add('hidden'));
    ui.newPatientForm.addEventListener('submit', handleAddPatient);

    // Upload Zone
    ui.dropZone.addEventListener('click', () => ui.fileInput.click());
    ui.dropZone.addEventListener('dragover', (e) => { e.preventDefault(); ui.dropZone.classList.add('dragover'); });
    ui.dropZone.addEventListener('dragleave', () => ui.dropZone.classList.remove('dragover'));
    ui.dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        ui.dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFileUpload(e.dataTransfer.files[0]);
    });
    ui.fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFileUpload(e.target.files[0]);
    });

    // Sample Image
    ui.useSampleBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        showToast('Loading sample X-ray...', 'success');
        try {
            const res = await fetch('sample_xray.png');
            const blob = await res.blob();
            const file = new File([blob], "sample_wrist.png", { type: "image/png" });
            handleFileUpload(file);
        } catch (e) {
            showToast('Failed to load sample image', 'error');
        }
    });

    ui.resetScanBtn.addEventListener('click', () => {
        ui.resultsContainer.classList.add('hidden');
        ui.dropZone.style.display = 'block';
        ui.fileInput.value = '';
    });
}

// --- HANDLERS ---
async function handleLogin(e) {
    e.preventDefault();
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    ui.loginBtnText.classList.add('hidden');
    ui.loginLoader.classList.remove('hidden');

    try {
        const formData = new FormData();
        formData.append('username', email);
        formData.append('password', password);

        const data = await fetchAPI('/auth/login', {
            method: 'POST',
            body: formData // Login endpoint uses OAuth2 password form
        });

        currentToken = data.access_token;
        localStorage.setItem('cortexray_token', currentToken);
        showToast('Login successful!');
        showDashboard();
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        ui.loginBtnText.classList.remove('hidden');
        ui.loginLoader.classList.add('hidden');
    }
}

function handleLogout() {
    currentToken = null;
    currentPatientId = null;
    localStorage.removeItem('cortexray_token');
    showLogin();
}

async function loadPatients() {
    try {
        const patients = await fetchAPI('/patients/');
        
        // Populate Table
        ui.patientsTableBody.innerHTML = '';
        patients.forEach(p => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${p.full_name}</strong></td>
                <td>${p.age}</td>
                <td>${p.gender}</td>
                <td>${p.comorbidities || 'None'}</td>
                <td>${new Date(p.created_at).toLocaleDateString()}</td>
            `;
            ui.patientsTableBody.appendChild(tr);
        });

        // Populate Dropdown
        ui.patientSelect.innerHTML = '<option value="">-- Select Patient --</option>';
        patients.forEach(p => {
            const opt = document.createElement('option');
            opt.value = p.id;
            opt.textContent = p.full_name;
            ui.patientSelect.appendChild(opt);
        });

        if(patients.length > 0 && !currentPatientId) {
            ui.patientSelect.value = patients[0].id;
            currentPatientId = patients[0].id;
        }

    } catch (err) {
        showToast('Failed to load patients', 'error');
    }
}

async function handleAddPatient(e) {
    e.preventDefault();
    const comorbStr = document.getElementById('p-comorb').value;
    const comorbidities = comorbStr ? comorbStr.split(',').map(s => s.trim()).filter(s => s) : [];

    const newPatient = {
        full_name: document.getElementById('p-name').value,
        age: parseInt(document.getElementById('p-age').value),
        gender: document.getElementById('p-gender').value,
        comorbidities: comorbidities
    };

    try {
        await fetchAPI('/patients/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newPatient)
        });
        showToast('Patient added successfully!');
        ui.newPatientModal.classList.add('hidden');
        ui.newPatientForm.reset();
        loadPatients();
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function handleFileUpload(file) {
    if (!currentPatientId) {
        showToast('Please select a patient first!', 'error');
        return;
    }

    ui.dropZone.style.display = 'none';
    ui.resultsContainer.classList.add('hidden');
    showToast('Analyzing X-ray... Please wait.', 'success');

    // Create local preview immediately
    const reader = new FileReader();
    reader.onload = (e) => { ui.originalImg.src = e.target.result; };
    reader.readAsDataURL(file);

    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('patient_id', currentPatientId);

        const data = await fetchAPI('/scan/upload', {
            method: 'POST',
            body: formData
        });

        updateResultsUI(data);

    } catch (err) {
        showToast(err.message, 'error');
        ui.dropZone.style.display = 'block';
    }
}

function updateResultsUI(data) {
    ui.resultsContainer.classList.remove('hidden');
    showToast('Analysis Complete!', 'success');

    if (data.fracture_detected) {
        ui.fracStatus.textContent = 'Fracture Detected';
        ui.fracStatus.className = 'status-badge danger';
        ui.confBadge.style.background = 'var(--danger)';
    } else {
        ui.fracStatus.textContent = 'No Fracture Detected';
        ui.fracStatus.className = 'status-badge success';
        ui.confBadge.style.background = 'var(--success)';
    }

    if (data.heatmap_url) {
        ui.heatmapImg.src = API_URL + data.heatmap_url;
        ui.heatmapImg.style.filter = "none";
    } else {
        ui.heatmapImg.src = ui.originalImg.src; // Fallback
        ui.heatmapImg.style.filter = data.fracture_detected ? "sepia(1) hue-rotate(-50deg) saturate(3) brightness(0.8)" : "none";
    }

    ui.confBadge.textContent = `${data.fracture_confidence.toFixed(1)}%`;
    ui.valRegion.textContent = data.bone_region;
    ui.valConf.textContent = `${data.fracture_confidence.toFixed(1)}% (${data.confidence_flag})`;
    
    ui.valRecovery.textContent = (data.rest_weeks_min && data.rest_weeks_max) ? `${data.rest_weeks_min} - ${data.rest_weeks_max} Weeks` : 'N/A';
    
    let intervention = 'Observation';
    if (data.referral_flag) intervention = 'Surgical Referral';
    else if (data.plaster_required) intervention = 'Casting/Immobilization';
    ui.valIntervention.textContent = intervention;
    
    ui.valTreatment.textContent = data.cast_type ? `${data.cast_type}. ${data.weight_bearing_status} weight-bearing.` : 'No specific treatment plan required.';

    // Setup Download Report Button
    ui.downloadReportBtn.onclick = async (e) => {
        e.preventDefault();
        if (!data.report_url) {
            showToast('Report not generated', 'error');
            return;
        }
        try {
            showToast('Downloading report...', 'success');
            const blob = await fetchAPI(data.report_url, { responseType: 'blob' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `Clinical_Report_${data.scan_id}.pdf`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
        } catch (err) {
            showToast('Failed to download report: ' + err.message, 'error');
        }
    };
}

// Run!
init();
