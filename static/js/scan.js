document.addEventListener("DOMContentLoaded", async () => {
    requireAuth();
    await loadComponents();
    if (typeof createParticleBackground === 'function') {
        createParticleBackground(document.getElementById('particle-bg'), 800);
    }

    // Module colors & icons
    const moduleConfig = {
        'fracture': { color: '#60A5FA', icon: '<path d="M18.37 2.63a1 1 0 01.7.29l2.01 2.01a1 1 0 010 1.42L8.35 19.08a3 3 0 01-1.55.83l-3.28.82.82-3.28a3 3 0 01.83-1.55L18.08 2.92a1 1 0 01.29-.29z"/>' },
        'arthritis': { color: '#A78BFA', icon: '<circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/>' },
        'osteoporosis': { color: '#F472B6', icon: '<path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/>' },
        'tb': { color: '#FBBF24', icon: '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>' },
        'lung_nodule': { color: '#34D399', icon: '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 14.5c-2.49 0-4.5-2.01-4.5-4.5S9.51 7.5 12 7.5s4.5 2.01 4.5 4.5-2.01 4.5-4.5 4.5z"/>' },
        'brain_tumor': { color: '#F87171', icon: '<path d="M12 2C8.69 2 6 4.69 6 8c0 1.95 1.15 3.65 2.85 4.5l-1.55 6.22A2 2 0 0 0 9.24 21h5.52a2 2 0 0 0 1.94-2.28l-1.55-6.22C16.85 11.65 18 9.95 18 8c0-3.31-2.69-6-6-6z"/>' },
        'brain_hemorrhage': { color: '#FB923C', icon: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>' },
        'bone_age': { color: '#22D3EE', icon: '<path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>' },
        'retinopathy': { color: '#818CF8', icon: '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>' }
    };

    const mod = window.MODULE_NAME || 'fracture';
    const conf = moduleConfig[mod];
    if (conf) {
        document.getElementById('module-icon-container').innerHTML = `<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="${conf.color}" stroke-width="2">${conf.icon}</svg>`;
        document.getElementById('module-icon-container').style.background = conf.color.replace(')', ', 0.1)').replace('rgb', 'rgba').replace('#', '') ; // Fallback if needed, we can just use fixed rgba
        // To properly do hex to rgba:
        const hex = conf.color.replace('#', '');
        const r = parseInt(hex.substring(0, 2), 16);
        const g = parseInt(hex.substring(2, 4), 16);
        const b = parseInt(hex.substring(4, 6), 16);
        document.getElementById('module-icon-container').style.background = `rgba(${r}, ${g}, ${b}, 0.1)`;
        document.getElementById('module-icon-container').style.boxShadow = `0 0 20px rgba(${r}, ${g}, ${b}, 0.2)`;
    }

    let selectedPatientId = null;
    let selectedFile = null;
    let selectedGender = null;

    const analyzeBtn = document.getElementById('analyze-btn');

    function checkFormValidity() {
        let valid = selectedPatientId && selectedFile;
        if (window.MODULE_NAME === 'bone_age' && !selectedGender) valid = false;
        
        if (valid) {
            analyzeBtn.disabled = false;
            analyzeBtn.style.opacity = '1';
            analyzeBtn.style.cursor = 'pointer';
        } else {
            analyzeBtn.disabled = true;
            analyzeBtn.style.opacity = '0.4';
            analyzeBtn.style.cursor = 'not-allowed';
        }
    }

    // --- STEP 1: PATIENTS ---
    const searchInput = document.getElementById('patient-search');
    const dropdown = document.getElementById('patient-dropdown');
    
    searchInput.addEventListener('input', async (e) => {
        const query = e.target.value.toLowerCase();
        if (query.length < 2) {
            dropdown.classList.add('hidden');
            return;
        }
        
        try {
            const res = await apiFetch('/patients/');
            if (res && res.ok) {
                const patients = await res.json();
                const filtered = patients.filter(p => p.full_name.toLowerCase().includes(query) || p.id.includes(query));
                
                dropdown.innerHTML = '';
                if (filtered.length > 0) {
                    filtered.forEach(p => {
                        const div = document.createElement('div');
                        div.className = 'p-3 hover:bg-white/10 cursor-pointer border-b border-white/5 transition';
                        div.innerHTML = `<div class="font-semibold text-white text-sm">${p.full_name}</div><div class="text-xs text-slate-400">ID: ${p.id.slice(0,8)} | Age: ${p.age} | ${p.gender}</div>`;
                        div.onclick = () => selectPatient(p.id, p.full_name, p.age, p.gender);
                        dropdown.appendChild(div);
                    });
                } else {
                    dropdown.innerHTML = '<div class="p-4 text-sm text-slate-400 text-center">No patients found</div>';
                }
                dropdown.classList.remove('hidden');
            }
        } catch (err) {
            console.error(err);
        }
    });

    document.addEventListener('click', (e) => {
        if (e.target !== searchInput && !dropdown.contains(e.target)) {
            dropdown.classList.add('hidden');
        }
    });

    window.selectPatient = function(id, name, age, gender) {
        selectedPatientId = id;
        document.getElementById('patient-name-display').textContent = name;
        document.getElementById('patient-id-display').textContent = `ID: ${id.slice(0,8)} | Age: ${age} | ${gender}`;
        document.getElementById('selected-patient').classList.remove('hidden');
        
        searchInput.value = '';
        dropdown.classList.add('hidden');
        document.getElementById('patient-search-container').classList.add('hidden');
        document.getElementById('or-divider').classList.add('hidden');
        document.getElementById('new-patient-form').classList.add('hidden');
        
        checkFormValidity();
    }

    window.clearPatient = function() {
        selectedPatientId = null;
        document.getElementById('selected-patient').classList.add('hidden');
        document.getElementById('patient-search-container').classList.remove('hidden');
        document.getElementById('or-divider').classList.remove('hidden');
        document.getElementById('new-patient-form').classList.remove('hidden');
        checkFormValidity();
    }

    window.createPatient = async function() {
        const name = document.getElementById('new-name').value;
        const age = document.getElementById('new-age').value;
        const gender = document.getElementById('new-gender').value;

        if (!name || !age || !gender) {
            showToast('Please fill all required patient fields', 'warning');
            return;
        }

        try {
            const res = await apiFetch('/patients/', {
                method: 'POST',
                body: JSON.stringify({
                    full_name: name,
                    age: parseInt(age),
                    gender: gender,
                    comorbidities: []
                })
            });

            const data = await res.json();
            if (res.ok) {
                showToast('Patient created successfully', 'success');
                selectPatient(data.id, data.full_name, data.age, data.gender);
            } else {
                throw new Error(data.detail || 'Failed to create patient');
            }
        } catch (e) {
            showToast(e.message, 'error');
        }
    }


    // --- STEP 2: IMAGE UPLOAD ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const dropContent = document.getElementById('drop-content');
    const preview = document.getElementById('file-preview');

    dropZone.addEventListener('dragover', (e) => { 
        e.preventDefault(); 
        dropZone.style.borderColor = '#14B8A6'; 
        dropZone.style.background = 'rgba(20,184,166,0.05)';
    });
    dropZone.addEventListener('dragleave', (e) => { 
        e.preventDefault(); 
        dropZone.style.borderColor = 'rgba(255,255,255,0.2)'; 
        dropZone.style.background = 'transparent';
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'rgba(255,255,255,0.2)';
        dropZone.style.background = 'transparent';
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) handleFile(fileInput.files[0]);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/') && !file.name.endsWith('.dcm')) {
            showToast('Please upload an image file', 'error');
            return;
        }
        selectedFile = file;
        document.getElementById('file-name').textContent = file.name;
        document.getElementById('file-size').textContent = (file.size / 1024 / 1024).toFixed(2) + ' MB';
        
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => document.getElementById('preview-img').src = e.target.result;
            reader.readAsDataURL(file);
        }

        dropContent.classList.add('hidden');
        preview.classList.remove('hidden');
        gsap.from(preview, { opacity: 0, scale: 0.9, duration: 0.3 });
        
        checkFormValidity();
    }

    window.clearFile = function() {
        selectedFile = null;
        fileInput.value = '';
        preview.classList.add('hidden');
        dropContent.classList.remove('hidden');
        checkFormValidity();
    }

    // --- STEP 3: BONE AGE GENDER ---
    if (window.MODULE_NAME === 'bone_age') {
        window.selectGender = function(val) {
            selectedGender = val === 'male' ? 'M' : 'F';
            const btnM = document.getElementById('gender-m');
            const btnF = document.getElementById('gender-f');
            
            btnM.style.borderColor = val === 'male' ? 'rgba(34,211,238,0.5)' : 'rgba(255,255,255,0.08)';
            btnM.style.background = val === 'male' ? 'rgba(34,211,238,0.1)' : 'rgba(255,255,255,0.03)';
            btnM.style.color = val === 'male' ? '#22D3EE' : '#94A3B8';
            
            btnF.style.borderColor = val === 'female' ? 'rgba(34,211,238,0.5)' : 'rgba(255,255,255,0.08)';
            btnF.style.background = val === 'female' ? 'rgba(34,211,238,0.1)' : 'rgba(255,255,255,0.03)';
            btnF.style.color = val === 'female' ? '#22D3EE' : '#94A3B8';
            
            checkFormValidity();
        }
    }

    // --- SUBMIT / ANALYSIS ---
    window.submitScan = async function() {
        if (analyzeBtn.disabled) return;

        showAnalysisProgress();

        const formData = new FormData();
        formData.append('patient_id', selectedPatientId);
        formData.append('file', selectedFile);
        if (window.MODULE_NAME === 'bone_age') formData.append('gender', selectedGender);

        try {
            const res = await apiFetch(`/${window.MODULE_NAME}/scan/upload`, {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if (res.ok) {
                setTimeout(() => {
                    window.location.href = `/results/${window.MODULE_NAME}/${data.scan_id}`;
                }, 3500); // Give the fake animation time to finish
            } else {
                throw new Error(data.detail || 'Analysis failed');
            }
        } catch (e) {
            document.getElementById('analysis-overlay').style.display = 'none';
            showToast(e.message, 'error');
        }
    }

    async function showAnalysisProgress() {
        document.getElementById('analysis-overlay').style.display = 'flex';

        const steps = [
            { text: 'Image uploaded and verified', delay: 400 },
            { text: 'Preprocessing with CLAHE', delay: 800 },
            { text: 'Running ResNet-50 inference...', delay: 1500 },
            { text: 'Generating Grad-CAM heatmap', delay: 800 },
            { text: 'Analysis complete!', delay: 500 }
        ];

        const container = document.getElementById('analysis-steps');
        container.innerHTML = '';

        for (let i = 0; i < steps.length; i++) {
            await new Promise(r => setTimeout(r, steps[i].delay));

            const step = document.createElement('div');
            step.className = 'flex items-center gap-3 mb-3';
            step.style.opacity = '0';

            if (i < steps.length - 1) {
                step.innerHTML = `
                    <div style="width:24px; height:24px; border-radius:50%; background:rgba(52,211,153,0.15); display:flex; align-items:center; justify-content:center;">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#34D399" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>
                    </div>
                    <span style="font-size:14px; color:#94A3B8;">${steps[i].text}</span>
                `;
            } else {
                step.innerHTML = `
                    <div style="width:24px; height:24px; border-radius:50%; background:rgba(20,184,166,0.2); display:flex; align-items:center; justify-content:center;">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#14B8A6" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>
                    </div>
                    <span style="font-size:14px; color:#14B8A6; font-weight:600;">${steps[i].text}</span>
                `;
            }

            container.appendChild(step);
            gsap.to(step, { opacity: 1, x: 0, duration: 0.3, ease: 'power2.out' });
            gsap.from(step, { x: -10, duration: 0.3, ease: 'power2.out' });
        }
    }
});
