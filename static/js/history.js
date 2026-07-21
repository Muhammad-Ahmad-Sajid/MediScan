document.addEventListener("DOMContentLoaded", async () => {
    requireAuth();
    await loadComponents();
    if (typeof createParticleBackground === 'function') {
        createParticleBackground(document.getElementById('particle-bg'), 800);
    }

    const container = document.getElementById('scans-container');
    const searchInput = document.getElementById('filter-search');
    const tabs = document.querySelectorAll('.history-tab');
    
    let allPatients = [];
    let allScans = [];
    let patientMap = {};
    let activeModule = 'all';

    // --- FETCH DATA ---
    try {
        const [patientsRes, scansRes] = await Promise.all([
            apiFetch('/patients/'),
            apiFetch('/patients/all_scans')
        ]);
        
        if (patientsRes.ok && scansRes.ok) {
            allPatients = await patientsRes.json();
            allScans = await scansRes.json();
            
            // Map for quick lookup
            allPatients.forEach(p => { patientMap[p.id] = p; });
            
            renderScans();
        } else {
            throw new Error('Failed to load data');
        }
    } catch (e) {
        container.innerHTML = '<div class="col-span-full text-center text-red-400 p-8 glass rounded-xl">Error loading history data.</div>';
        console.error(e);
    }

    // --- TABS LOGIC ---
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            activeModule = tab.dataset.module;
            renderScans();
        });
    });

    searchInput.addEventListener('input', () => renderScans());

    // --- RENDER SCANS ---
    function renderScans() {
        container.innerHTML = '';
        const query = searchInput.value.toLowerCase();
        
        let filtered = allScans;
        
        // Filter by module
        if (activeModule !== 'all') {
            filtered = filtered.filter(s => s.module === activeModule);
        }
        
        // Filter by search
        if (query) {
            filtered = filtered.filter(s => {
                const p = patientMap[s.patient_id];
                const pName = p ? p.full_name.toLowerCase() : '';
                return s.patient_id.toLowerCase().includes(query) || pName.includes(query) || s.scan_id.toLowerCase().includes(query);
            });
        }

        // Sort descending by timestamp
        filtered.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

        if (filtered.length === 0) {
            container.innerHTML = '<div class="col-span-full text-center text-slate-400 p-12 glass rounded-xl">No scans found matching the criteria.</div>';
            return;
        }

        filtered.forEach((scan, i) => {
            const p = patientMap[scan.patient_id] || { full_name: 'Unknown', id: scan.patient_id };
            
            // Module badge color mapping
            const colorMap = {
                'fracture': 'blue', 'arthritis': 'violet', 'osteoporosis': 'pink',
                'tb': 'amber', 'lung_nodule': 'emerald', 'brain_tumor': 'red',
                'brain_hemorrhage': 'orange', 'bone_age': 'cyan', 'retinopathy': 'indigo'
            };
            const col = colorMap[scan.module] || 'teal';
            const dateStr = new Date(scan.timestamp).toLocaleDateString();

            const card = document.createElement('div');
            card.className = 'glass tilt-card relative overflow-hidden cursor-pointer';
            card.style.padding = '24px';
            card.style.borderRadius = '16px';
            card.innerHTML = `
                <div style="position:absolute; top:0; left:0; right:0; height:3px; background:var(--tw-colors-${col}-400, #14B8A6);"></div>
                
                <div class="flex justify-between items-start mb-4">
                    <div style="font-size:11px; padding:4px 8px; border-radius:6px; border:1px solid var(--tw-colors-${col}-500, #14B8A6); background:rgba(255,255,255,0.05); color:var(--tw-colors-${col}-400, #14B8A6); text-transform:uppercase; letter-spacing:1px; display:inline-block;">
                        ${scan.module.replace('_', ' ')}
                    </div>
                    <span style="font-size:12px; color:#64748B;">${dateStr}</span>
                </div>
                
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
                    <div style="width:40px; height:40px; border-radius:10px; background:rgba(255,255,255,0.05); display:flex; align-items:center; justify-content:center; font-weight:600; font-size:14px; color:#F1F5F9;">
                        ${p.full_name.substring(0,2).toUpperCase()}
                    </div>
                    <div>
                        <h4 style="font-size:15px; font-weight:600; color:#F1F5F9; margin:0 0 2px;">${p.full_name}</h4>
                        <p style="font-size:12px; color:#64748B; margin:0;">ID: ${p.id.slice(0,8)}</p>
                    </div>
                </div>

                <div style="padding-top:16px; border-top:1px solid rgba(255,255,255,0.06); display:flex; align-items:center; justify-content:space-between;">
                    <span style="font-size:13px; color:#94A3B8;" class="truncate pr-2">${scan.result || 'Analysis complete'}</span>
                    <span style="font-size:12px; font-weight:500; color:#14B8A6;">View Results &rarr;</span>
                </div>
            `;
            
            card.onclick = () => { window.location.href = `/results/${scan.module}/${scan.scan_id}`; };

            // Hover effect
            card.addEventListener('mouseover', () => { card.style.boxShadow = `0 0 20px var(--tw-colors-${col}-500, rgba(20,184,166,0.2))`; });
            card.addEventListener('mouseout', () => { card.style.boxShadow = 'none'; });

            gsap.from(card, {
                opacity: 0, y: 20, duration: 0.5, delay: i * 0.05, ease: 'power2.out'
            });

            container.appendChild(card);
        });
        
        initTiltCards(); // Re-init tilt for newly rendered cards
    }

    // --- MODAL (Optional if user wants to click patient to see details instead of going straight to results, but currently card clicks go to results) ---
    window.closePatientModal = function() {
        document.getElementById('patient-modal').classList.add('hidden');
    }
});
