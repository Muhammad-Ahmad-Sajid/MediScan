requireAuth();
loadComponents();

// Define module metadata here too for frontend rendering
const modMeta = {
    'fracture': { name: 'Bone Fracture Detection', color: '#3B82F6' },
    'arthritis': { name: 'Arthritis Grading', color: '#8B5CF6' },
    'osteoporosis': { name: 'Osteoporosis Screening', color: '#EC4899' },
    'tb': { name: 'TB Screening', color: '#F59E0B' },
    'lung-nodule': { name: 'Lung Nodule Detection', color: '#10B981' },
    'brain-tumor': { name: 'Brain Tumor Classification', color: '#EF4444' },
    'brain-hemorrhage': { name: 'Brain Hemorrhage', color: '#DC2626' },
    'bone-age': { name: 'Bone Age Estimation', color: '#06B6D4' },
    'retinopathy': { name: 'Diabetic Retinopathy', color: '#6366F1' }
};

const modData = modMeta[currentModule];
if(modData) {
    document.getElementById('breadcrumb-mod').textContent = modData.name;
    document.getElementById('mod-title').textContent = modData.name;
    document.getElementById('mod-icon').src = `/static/img/module-icons/${currentModule}.svg`;
    document.getElementById('mod-icon-bg').style.backgroundColor = modData.color + '20'; // 20% opacity
}

// Module specific fields
const specContainer = document.getElementById('module-specific-container');
const specFields = document.getElementById('module-specific-fields');
if (currentModule === 'bone-age') {
    specContainer.classList.remove('hidden');
    specFields.innerHTML = `
        <label class="block text-sm font-medium text-slate-700 mb-1">Patient Gender (Required for Bone Age)</label>
        <div class="flex space-x-4">
            <label class="inline-flex items-center"><input type="radio" name="gender" value="Male" class="text-primary-600 focus:ring-primary-400" checked> <span class="ml-2 text-slate-700">Male</span></label>
            <label class="inline-flex items-center"><input type="radio" name="gender" value="Female" class="text-primary-600 focus:ring-primary-400"> <span class="ml-2 text-slate-700">Female</span></label>
        </div>
    `;
} else if (currentModule === 'fracture') {
    specContainer.classList.remove('hidden');
    specFields.innerHTML = `
        <label class="block text-sm font-medium text-slate-700 mb-1">Clinical Notes (Optional)</label>
        <textarea id="clinical-notes" rows="2" class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-400 outline-none text-sm"></textarea>
    `;
}

// Load Patients
async function loadPatients() {
    try {
        const res = await apiFetch('/patients/');
        if(res && res.ok) {
            const patients = await res.json();
            const select = document.getElementById('patient-select');
            patients.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p.id;
                opt.textContent = `${p.full_name} (${p.id})`;
                select.appendChild(opt);
            });
        }
    } catch(err) {
        console.error(err);
    }
}
document.addEventListener('DOMContentLoaded', loadPatients);

// Create Patient
document.getElementById('btn-create-patient').addEventListener('click', async () => {
    const name = document.getElementById('new-pat-name').value;
    const age = parseInt(document.getElementById('new-pat-age').value);
    const gender = document.getElementById('new-pat-gender').value;
    const pid = document.getElementById('new-pat-id').value;
    
    if(!name || !age) {
        showToast('Please provide Name and Age', 'error');
        return;
    }
    
    try {
        const payload = { full_name: name, age: age, gender: gender };
        if(pid) payload.id = pid;
        
        const res = await apiFetch('/patients/', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
        
        if(res && res.ok) {
            const p = await res.json();
            const select = document.getElementById('patient-select');
            const opt = document.createElement('option');
            opt.value = p.id;
            opt.textContent = `${p.full_name} (${p.id})`;
            select.appendChild(opt);
            select.value = p.id;
            showToast('Patient created successfully');
            document.getElementById('new-patient-form').reset();
        } else {
            const data = await res.json();
            showToast(data.detail || 'Failed to create patient', 'error');
        }
    } catch(err) {
        showToast('Connection error', 'error');
    }
});

// Drag and Drop Logic
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const placeholder = document.getElementById('drop-placeholder');
const previewCont = document.getElementById('preview-container');
const imgPreview = document.getElementById('img-preview');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const btnRemove = document.getElementById('btn-remove-file');

let selectedFile = null;

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});
function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});
['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    if(dt.files && dt.files.length) handleFile(dt.files[0]);
});

fileInput.addEventListener('change', function() {
    if(this.files && this.files.length) handleFile(this.files[0]);
});

function handleFile(file) {
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = (file.size / (1024*1024)).toFixed(2) + ' MB';
    
    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => { imgPreview.src = reader.result; };
    } else {
        imgPreview.src = '/static/img/logo.svg'; // fallback for dicom
    }
    
    placeholder.classList.add('hidden');
    previewCont.classList.remove('hidden');
}

btnRemove.addEventListener('click', (e) => {
    e.stopPropagation(); // prevent clicking dropzone
    selectedFile = null;
    fileInput.value = '';
    previewCont.classList.add('hidden');
    placeholder.classList.remove('hidden');
});

// Submit Analysis
document.getElementById('btn-analyze').addEventListener('click', async () => {
    const patientId = document.getElementById('patient-select').value;
    if(!patientId) { showToast('Please select a patient', 'error'); return; }
    if(!selectedFile) { showToast('Please upload an image', 'error'); return; }

    // Show Progress
    const overlay = document.getElementById('progress-overlay');
    document.getElementById('progress-title').textContent = `Analyzing ${modData ? modData.name : 'Scan'}...`;
    overlay.classList.remove('hidden');
    
    try {
        const formData = new FormData();
        formData.append('patient_id', patientId);
        formData.append('file', selectedFile);
        
        if (currentModule === 'bone-age') {
            const gender = document.querySelector('input[name="gender"]:checked').value;
            formData.append('gender', gender);
        }
        
        const res = await apiFetch(`/${currentModule}/scan/upload`, {
            method: 'POST',
            body: formData
        });
        
        if(res && res.ok) {
            const data = await res.json();
            // Step UI updates simulation
            document.getElementById('step-1').classList.replace('text-slate-400', 'text-emerald-600');
            setTimeout(() => {
                document.getElementById('step-2').classList.replace('text-slate-400', 'text-emerald-600');
            }, 800);
            setTimeout(() => {
                window.location.href = `/results/${currentModule}/${data.id}`;
            }, 1500);
        } else {
            overlay.classList.add('hidden');
            const data = await res.json();
            showToast(data.detail || 'Analysis failed', 'error');
        }
    } catch(err) {
        overlay.classList.add('hidden');
        showToast('Connection error during analysis', 'error');
    }
});
