requireAuth();
loadComponents();

const modMeta = [
    { id: 'all', name: 'All Scans', color: '#64748B' },
    { id: 'fracture', name: 'Fracture', color: '#3B82F6' },
    { id: 'arthritis', name: 'Arthritis', color: '#8B5CF6' },
    { id: 'osteoporosis', name: 'Osteoporosis', color: '#EC4899' },
    { id: 'tb', name: 'TB', color: '#F59E0B' },
    { id: 'lung-nodule', name: 'Lung Nodule', color: '#10B981' },
    { id: 'brain-tumor', name: 'Brain Tumor', color: '#EF4444' },
    { id: 'brain-hemorrhage', name: 'Brain Hemorrhage', color: '#DC2626' },
    { id: 'bone-age', name: 'Bone Age', color: '#06B6D4' },
    { id: 'retinopathy', name: 'Retinopathy', color: '#6366F1' }
];

let currentPatientId = new URLSearchParams(window.location.search).get('patient');
let activeTab = 'all';
let viewMode = 'list'; // 'list' or 'grid'
let patientScans = [];

// Load patients for select dropdown
apiFetch('/patients/').then(r=>r.json()).then(pts => {
    const sel = document.getElementById('patient-select');
    pts.forEach(p => {
        const o = document.createElement('option');
        o.value = p.id;
        o.textContent = `${p.full_name} (${p.id})`;
        sel.appendChild(o);
    });
    if(currentPatientId) {
        sel.value = currentPatientId;
        loadPatientData(currentPatientId);
    }
});

document.getElementById('patient-search-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const pid = document.getElementById('patient-select').value;
    if(pid) {
        currentPatientId = pid;
        const url = new URL(window.location);
        url.searchParams.set('patient', pid);
        window.history.pushState({}, '', url);
        loadPatientData(pid);
    }
});

async function loadPatientData(pid) {
    try {
        const res = await apiFetch(`/patients/${pid}`);
        if(res && res.ok) {
            const pt = await res.json();
            document.getElementById('pat-name').textContent = pt.full_name;
            document.getElementById('pat-id').textContent = pt.id;
            document.getElementById('pat-age').textContent = pt.age;
            document.getElementById('pat-gender').textContent = pt.gender;
            
            const initials = pt.full_name.split(' ').map(n=>n[0]).join('').substring(0,2).toUpperCase();
            document.getElementById('pat-initials').textContent = initials;
            
            document.getElementById('patient-info-card').classList.remove('hidden');
            document.getElementById('history-area').classList.remove('hidden');
            
            await fetchAllScans(pid);
        } else {
            showToast('Patient not found', 'error');
        }
    } catch(err) {
        showToast('Error loading patient', 'error');
    }
}

async function fetchAllScans(pid) {
    try {
        const res = await apiFetch(`/patients/${pid}/scans`);
        if(res && res.ok) {
            patientScans = await res.json();
            
            document.getElementById('pat-total-scans').textContent = patientScans.length;
            if(patientScans.length > 0) {
                const latest = new Date(patientScans[0].upload_timestamp);
                document.getElementById('pat-last-scan').textContent = latest.toLocaleDateString();
            } else {
                document.getElementById('pat-last-scan').textContent = 'N/A';
            }
            
            renderTabs();
            renderScans();
        }
    } catch (e) {
        showToast('Failed to load scan history', 'error');
    }
}

function renderTabs() {
    const tabsCont = document.getElementById('module-tabs');
    tabsCont.innerHTML = '';
    
    modMeta.forEach(mod => {
        let count = patientScans.length;
        if(mod.id !== 'all') {
            count = patientScans.filter(s => s.module === mod.id).length;
        }
        
        const btn = document.createElement('button');
        btn.className = `whitespace-nowrap px-4 py-2 text-sm font-medium rounded-full transition-colors ${activeTab === mod.id ? 'bg-primary-700 text-white shadow-sm' : 'bg-white border border-slate-200 text-slate-600 hover:bg-slate-50'}`;
        btn.innerHTML = `${mod.name} <span class="ml-1 px-1.5 py-0.5 rounded-full text-xs ${activeTab === mod.id ? 'bg-primary-800 text-primary-100' : 'bg-slate-100 text-slate-500'}">${count}</span>`;
        btn.onclick = () => {
            activeTab = mod.id;
            renderTabs();
            renderScans();
        };
        tabsCont.appendChild(btn);
    });
}

function getBadgeColor(text) {
    const t = String(text).toLowerCase();
    if(t.includes('normal') || t.includes('grade 0')) return 'bg-emerald-100 text-emerald-700';
    if(t.includes('fracture') || t.includes('detected') || t.includes('hemorrhage') || t.includes('glioma')) return 'bg-red-100 text-red-700';
    if(t.includes('grade 4') || t.includes('grade 3')) return 'bg-orange-100 text-orange-700';
    if(t.includes('grade 1') || t.includes('grade 2') || t.includes('meningioma')) return 'bg-amber-100 text-amber-700';
    if(t.includes('age')) return 'bg-teal-100 text-teal-700';
    return 'bg-slate-100 text-slate-700';
}

function renderScans() {
    const cont = document.getElementById('scans-container');
    const empty = document.getElementById('empty-state');
    cont.innerHTML = '';
    
    const filtered = activeTab === 'all' ? patientScans : patientScans.filter(s => s.module === activeTab);
    
    if(filtered.length === 0) {
        cont.classList.add('hidden');
        empty.classList.remove('hidden');
        return;
    }
    
    empty.classList.add('hidden');
    cont.classList.remove('hidden');
    
    if (viewMode === 'list') {
        cont.className = 'space-y-4';
    } else {
        cont.className = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6';
    }

    filtered.forEach(scan => {
        const modInfo = modMeta.find(m => m.id === scan.module) || modMeta[0];
        let diagText = scan.diagnosis;
        if(scan.module === 'bone-age') diagText = `Est. Age: ${(scan.diagnosis/12).toFixed(1)} yrs`;
        if(scan.module === 'arthritis' || scan.module === 'retinopathy') diagText = `Grade ${scan.diagnosis}`;
        
        const badgeCls = getBadgeColor(diagText);
        let confText = scan.module === 'bone-age' ? 'Regression' : `Conf: ${Math.round(scan.confidence * 100)}%`;
        const dateStr = new Date(scan.upload_timestamp).toLocaleString();
        
        const origImg = API_BASE + '/' + scan.file_path;
        const heatImg = scan.heatmap_path ? (API_BASE + '/' + scan.heatmap_path) : origImg;

        if (viewMode === 'list') {
            cont.insertAdjacentHTML('beforeend', `
                <div class="bg-surface rounded-xl border border-slate-200 p-4 flex flex-col sm:flex-row items-center gap-4 hover:shadow-md transition-shadow">
                    <div class="relative w-20 h-20 bg-slate-900 rounded-lg overflow-hidden shrink-0 group cursor-pointer" onclick="document.getElementById('lightbox-img').src='${origImg}'; document.getElementById('lightbox').classList.remove('hidden')">
                        <img src="${origImg}" class="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity">
                        <div class="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                            <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7"></path></svg>
                        </div>
                    </div>
                    <div class="flex-1 text-center sm:text-left">
                        <div class="flex items-center justify-center sm:justify-start gap-2 mb-1">
                            <h4 class="font-bold text-slate-800">${modInfo.name}</h4>
                            <span class="text-xs text-slate-400">&bull; ${dateStr}</span>
                        </div>
                        <div class="flex flex-wrap items-center justify-center sm:justify-start gap-2 mb-2">
                            <span class="px-3 py-1 rounded-full text-xs font-bold ${badgeCls}">${String(diagText).toUpperCase()}</span>
                            <span class="text-xs font-medium text-slate-500">${confText}</span>
                        </div>
                        <p class="text-xs text-slate-500 truncate max-w-md">${scan.recommendation || 'No recommendation'}</p>
                    </div>
                    <div class="flex flex-col gap-2 shrink-0 w-full sm:w-auto mt-2 sm:mt-0">
                        <a href="/results/${scan.module}/${scan.id}" class="w-full px-4 py-2 bg-primary-50 text-primary-700 hover:bg-primary-100 rounded-lg text-sm font-medium text-center transition-colors">View Details</a>
                        <button onclick="downloadReport('${scan.module}', '${scan.id}')" class="w-full px-4 py-2 border border-slate-200 text-slate-600 hover:bg-slate-50 rounded-lg text-sm font-medium transition-colors flex items-center justify-center">
                            <svg class="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg> PDF
                        </button>
                    </div>
                </div>
            `);
        } else {
            cont.insertAdjacentHTML('beforeend', `
                <div class="bg-surface rounded-xl border border-slate-200 overflow-hidden hover:shadow-md transition-shadow flex flex-col">
                    <div class="relative h-40 bg-slate-900 group cursor-pointer" onclick="document.getElementById('lightbox-img').src='${origImg}'; document.getElementById('lightbox').classList.remove('hidden')">
                        <img src="${origImg}" class="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity">
                        <div class="absolute top-2 left-2 px-2 py-1 bg-black/60 backdrop-blur rounded text-xs text-white font-medium">${modInfo.name}</div>
                    </div>
                    <div class="p-4 flex-1 flex flex-col">
                        <div class="flex justify-between items-start mb-2">
                            <span class="px-2 py-1 rounded-full text-xs font-bold ${badgeCls}">${String(diagText).toUpperCase()}</span>
                            <span class="text-xs text-slate-400 text-right">${new Date(scan.upload_timestamp).toLocaleDateString()}</span>
                        </div>
                        <p class="text-xs font-medium text-slate-500 mb-2">${confText}</p>
                        <p class="text-xs text-slate-500 line-clamp-2 mb-4 flex-1">${scan.recommendation || 'No recommendation'}</p>
                        
                        <div class="flex gap-2 mt-auto">
                            <a href="/results/${scan.module}/${scan.id}" class="flex-1 py-1.5 bg-primary-50 text-primary-700 hover:bg-primary-100 rounded border border-primary-100 text-xs font-medium text-center transition-colors">Details</a>
                            <button onclick="downloadReport('${scan.module}', '${scan.id}')" class="flex-1 py-1.5 border border-slate-200 text-slate-600 hover:bg-slate-50 rounded text-xs font-medium transition-colors flex items-center justify-center">
                                PDF
                            </button>
                        </div>
                    </div>
                </div>
            `);
        }
    });
}

document.getElementById('view-list').addEventListener('click', () => { viewMode = 'list'; updateViewBtns(); renderScans(); });
document.getElementById('view-grid').addEventListener('click', () => { viewMode = 'grid'; updateViewBtns(); renderScans(); });

function updateViewBtns() {
    const listBtn = document.getElementById('view-list');
    const gridBtn = document.getElementById('view-grid');
    if(viewMode === 'list') {
        listBtn.classList.replace('text-slate-500', 'text-slate-800'); listBtn.classList.replace('hover:text-slate-800', 'bg-white'); listBtn.classList.add('shadow-sm');
        gridBtn.classList.replace('text-slate-800', 'text-slate-500'); gridBtn.classList.replace('bg-white', 'hover:text-slate-800'); gridBtn.classList.remove('shadow-sm');
    } else {
        gridBtn.classList.replace('text-slate-500', 'text-slate-800'); gridBtn.classList.replace('hover:text-slate-800', 'bg-white'); gridBtn.classList.add('shadow-sm');
        listBtn.classList.replace('text-slate-800', 'text-slate-500'); listBtn.classList.replace('bg-white', 'hover:text-slate-800'); listBtn.classList.remove('shadow-sm');
    }
}

async function downloadReport(module, scanId) {
    try {
        const res = await apiFetch(`/${module}/scan/${scanId}/report`);
        if(res && res.ok) {
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `MediScan_${module.toUpperCase()}_Report_${scanId}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
        } else {
            showToast('Failed to generate report', 'error');
        }
    } catch(err) {
        showToast('Connection error', 'error');
    }
}
