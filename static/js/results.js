requireAuth();
loadComponents();

const modMeta = {
    'fracture': { name: 'Bone Fracture Detection', color: '#3B82F6', opt: ['Fractured', 'Normal'] },
    'arthritis': { name: 'Arthritis Grading', color: '#8B5CF6', opt: ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'] },
    'osteoporosis': { name: 'Osteoporosis Screening', color: '#EC4899', opt: ['Osteoporosis', 'Normal'] },
    'tb': { name: 'TB Screening', color: '#F59E0B', opt: ['TB Detected', 'Normal'] },
    'lung-nodule': { name: 'Lung Nodule Detection', color: '#10B981', opt: ['Nodule Detected', 'Normal'] },
    'brain-tumor': { name: 'Brain Tumor Classification', color: '#EF4444', opt: ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor'] },
    'brain-hemorrhage': { name: 'Brain Hemorrhage', color: '#DC2626', opt: ['Hemorrhage Detected', 'Normal'] },
    'bone-age': { name: 'Bone Age Estimation', color: '#06B6D4', opt: [] }, // Regression
    'retinopathy': { name: 'Diabetic Retinopathy', color: '#6366F1', opt: ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'] }
};

const modData = modMeta[currentModule];
if(modData) {
    document.getElementById('breadcrumb-mod').textContent = modData.name;
    document.getElementById('mod-title').textContent = modData.name;
    document.getElementById('mod-icon').src = `/static/img/module-icons/${currentModule}.svg`;
    document.getElementById('mod-icon-bg').style.backgroundColor = modData.color + '20';
}

// Set up Override Options
const ovSelect = document.getElementById('override-value');
if(modData && modData.opt.length > 0) {
    modData.opt.forEach(opt => {
        const el = document.createElement('option');
        el.value = opt; el.textContent = opt;
        ovSelect.appendChild(el);
    });
} else {
    // Free text for regression or others without fixed classes
    ovSelect.outerHTML = '<input type="text" id="override-value" required class="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary-400 outline-none" placeholder="Enter corrected diagnosis">';
}

// Lightbox
function openLightbox(src) {
    if(!src) return;
    document.getElementById('lightbox-img').src = src;
    document.getElementById('lightbox').classList.remove('hidden');
}
function closeLightbox() { document.getElementById('lightbox').classList.add('hidden'); }

// Fetch Results
async function loadResults() {
    try {
        const res = await apiFetch(`/${currentModule}/scan/${currentScanId}`);
        if(res && res.ok) {
            const data = await res.json();
            renderResults(data);
        } else {
            showToast('Failed to load scan details', 'error');
        }
    } catch(err) {
        showToast('Connection error', 'error');
    }
}

function renderResults(data) {
    // Images
    const imgOrigUrl = API_BASE + '/' + data.file_path;
    const imgHeatUrl = data.heatmap_path ? (API_BASE + '/' + data.heatmap_path) : imgOrigUrl;
    
    document.getElementById('img-display').src = imgOrigUrl;
    document.getElementById('img-sbs-orig').src = imgOrigUrl;
    document.getElementById('img-sbs-heat').src = imgHeatUrl;
    
    // Tab logic
    const tabOrig = document.getElementById('tab-original');
    const tabHeat = document.getElementById('tab-heatmap');
    const tabSbs = document.getElementById('tab-sidebyside');
    const viewSgl = document.getElementById('view-single');
    const viewSbs = document.getElementById('view-sidebyside');
    const imgDisp = document.getElementById('img-display');
    
    const activeClass = ['border-primary-700', 'text-primary-700', 'bg-white'];
    const inactiveClass = ['border-transparent', 'text-slate-500', 'bg-transparent'];
    
    function setTab(activeTab) {
        [tabOrig, tabHeat, tabSbs].forEach(t => {
            t.classList.remove(...activeClass);
            t.classList.add(...inactiveClass);
        });
        activeTab.classList.remove(...inactiveClass);
        activeTab.classList.add(...activeClass);
        
        if(activeTab === tabSbs) {
            viewSgl.classList.add('hidden');
            viewSbs.classList.remove('hidden');
            viewSbs.classList.add('flex');
        } else {
            viewSbs.classList.remove('flex', 'hidden');
            viewSbs.classList.add('hidden');
            viewSgl.classList.remove('hidden');
            if(activeTab === tabOrig) imgDisp.src = imgOrigUrl;
            if(activeTab === tabHeat) imgDisp.src = imgHeatUrl;
        }
    }
    
    tabOrig.onclick = () => setTab(tabOrig);
    tabHeat.onclick = () => setTab(tabHeat);
    tabSbs.onclick = () => setTab(tabSbs);
    
    // Header details
    if(data.patient_id) {
        document.getElementById('header-patient-id').textContent = data.patient_id;
        document.getElementById('btn-view-history').onclick = () => window.location.href = `/history?patient=${data.patient_id}`;
    }
    document.getElementById('scan-date').textContent = "Uploaded: " + (data.upload_timestamp ? new Date(data.upload_timestamp).toLocaleString() : 'N/A');
    
    // Patient details lookup
    apiFetch(`/patients/${data.patient_id}`).then(r => r.json()).then(p => {
        if(p && p.full_name) document.getElementById('header-patient-name').textContent = p.full_name;
    }).catch(()=>{});

    // Diagnosis & Confidence
    const badge = document.getElementById('diag-badge');
    const fill = document.getElementById('conf-fill');
    const confVal = document.getElementById('conf-val');
    const confLabel = document.getElementById('conf-label');
    
    // Try to normalize diagnosis to text and colors
    let diagText = data.diagnosis || data.result_class || data.predicted_class || 'Unknown';
    let conf = data.confidence || data.probability || data.confidence_score || 0;
    if(typeof conf === 'string') conf = parseFloat(conf.replace('%', ''));
    if(conf <= 1) conf = conf * 100; // if decimal
    
    // Custom handling for modules
    if(currentModule === 'bone-age') {
        diagText = `Estimated Age: ${data.bone_age_months ? (data.bone_age_months/12).toFixed(1) + ' yrs' : data.predicted_age_months}`;
        conf = 100; // Regression doesn't have classification confidence usually, just fill it
    } else if (currentModule === 'arthritis' || currentModule === 'retinopathy') {
        diagText = `Grade ${data.predicted_class}`;
    }

    badge.textContent = String(diagText).toUpperCase();
    
    // Set Badge Color based on diagnosis
    let colorClass = 'bg-slate-200 text-slate-700'; // default
    const dtL = String(diagText).toLowerCase();
    if(dtL.includes('normal') || dtL.includes('no ') || dtL.includes('grade 0')) {
        colorClass = 'bg-emerald-100 text-emerald-700 border border-emerald-200';
    } else if (dtL.includes('fracture') || dtL.includes('detected') || dtL.includes('hemorrhage') || dtL.includes('glioma')) {
        colorClass = 'bg-red-100 text-red-700 border border-red-200';
    } else if (dtL.includes('grade 4') || dtL.includes('grade 3')) {
        colorClass = 'bg-orange-100 text-orange-700 border border-orange-200';
    } else if (dtL.includes('grade 1') || dtL.includes('grade 2') || dtL.includes('meningioma')) {
        colorClass = 'bg-amber-100 text-amber-700 border border-amber-200';
    } else if (dtL.includes('age')) {
        colorClass = 'bg-teal-100 text-teal-700 border border-teal-200';
    }
    badge.className = `px-6 py-2 rounded-full text-lg font-bold shadow-sm ${colorClass}`;
    
    // Confidence bar
    if(currentModule !== 'bone-age') {
        confVal.textContent = Math.round(conf) + '%';
        fill.style.width = conf + '%';
        if(conf >= 85) { fill.classList.add('high'); confLabel.textContent = 'High Confidence'; }
        else if (conf >= 60) { fill.classList.add('medium'); confLabel.textContent = 'Moderate Confidence'; }
        else { fill.classList.add('low'); confLabel.textContent = 'Low Confidence'; }
    } else {
        confVal.textContent = 'N/A';
        fill.style.width = '100%';
        fill.classList.add('high');
        confLabel.textContent = 'Regression Model';
    }

    // Recommendation
    document.getElementById('recommendation-text').textContent = data.recommendation || 'No specific clinical recommendation provided by the model for this scan.';
    
    // Safety Flags
    const sc = document.getElementById('safety-card');
    const st = document.getElementById('safety-text');
    const sTitle = document.getElementById('safety-title');
    
    if(data.glioma_risk_flag) {
        sc.classList.remove('hidden');
        sc.classList.add('bg-red-50', 'border-red-500', 'text-red-800');
        sTitle.textContent = "Urgent: High-Risk Glioma";
        st.textContent = "Immediate neurosurgical consultation recommended.";
    } else if (data.referable_risk_flag) {
        sc.classList.remove('hidden');
        sc.classList.add('bg-amber-50', 'border-amber-500', 'text-amber-800');
        sTitle.textContent = "Referable Diabetic Retinopathy";
        st.textContent = "Patient requires ophthalmology referral.";
    } else if (currentModule === 'brain-hemorrhage') {
        // Info flag
        sc.classList.remove('hidden');
        sc.classList.add('bg-slate-50', 'border-slate-500', 'text-slate-800');
        sTitle.textContent = "Experimental Module";
        st.textContent = "Trained on a small dataset (CQ500). Use with extreme caution.";
    }

    // Prognosis
    if(currentModule === 'fracture') {
        apiFetch(`/prognosis/fracture/${data.id}`).then(r => {
            if(r.ok) return r.json();
            return null;
        }).then(prog => {
            if(prog) {
                const pc = document.getElementById('prognosis-card');
                const pl = document.getElementById('prognosis-list');
                pc.classList.remove('hidden');
                
                let srgCls = prog.surgery_referral ? 'text-red-600 font-bold' : 'text-slate-700';
                pl.innerHTML = `
                    <li class="flex justify-between"><span class="text-slate-500">Est. Healing Time</span> <span class="font-medium">${prog.healing_time_weeks} weeks</span></li>
                    <li class="flex justify-between"><span class="text-slate-500">Recommended Cast</span> <span class="font-medium">${prog.cast_type}</span></li>
                    <li class="flex justify-between"><span class="text-slate-500">Weight Bearing</span> <span class="font-medium">${prog.weight_bearing}</span></li>
                    <li class="flex justify-between border-t pt-2"><span class="text-slate-500">Surgery Referral</span> <span class="${srgCls}">${prog.surgery_referral ? 'YES' : 'NO'}</span></li>
                    <li class="flex justify-between"><span class="text-slate-500">Follow-up Interval</span> <span class="font-medium">${prog.follow_up_interval}</span></li>
                `;
            }
        }).catch(()=>{});
    }
}

// Buttons
document.getElementById('btn-new-scan').onclick = () => window.location.href = `/scan/${currentModule}`;

// Download PDF
document.getElementById('btn-download-pdf').addEventListener('click', async (e) => {
    const btn = e.currentTarget;
    const origHtml = btn.innerHTML;
    btn.innerHTML = '<svg class="w-5 h-5 mr-2 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg> Generating...';
    btn.disabled = true;

    try {
        const res = await apiFetch(`/${currentModule}/scan/${currentScanId}/report`);
        if(res && res.ok) {
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `MediScan_${currentModule.toUpperCase()}_Report_${currentScanId}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            showToast('Report downloaded successfully');
        } else {
            showToast('Failed to generate report', 'error');
        }
    } catch(err) {
        showToast('Connection error', 'error');
    } finally {
        btn.innerHTML = origHtml;
        btn.disabled = false;
    }
});

// Override Logic
document.getElementById('btn-override').onclick = () => document.getElementById('override-modal').classList.remove('hidden');

document.getElementById('override-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const ovVal = document.getElementById('override-value').value;
    const ovNotes = document.getElementById('override-notes').value;
    
    try {
        const res = await apiFetch(`/${currentModule}/${currentScanId}/override`, {
            method: 'PATCH',
            body: JSON.stringify({
                clinician_override: ovVal,
                override_notes: ovNotes
            })
        });
        
        if(res && res.ok) {
            showToast('Diagnosis override submitted successfully');
            document.getElementById('override-modal').classList.add('hidden');
            // Refresh results
            loadResults();
        } else {
            showToast('Failed to submit override', 'error');
        }
    } catch(err) {
        showToast('Connection error', 'error');
    }
});

document.addEventListener('DOMContentLoaded', loadResults);
