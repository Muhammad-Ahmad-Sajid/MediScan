requireAuth();
loadComponents();

const modules = [
    { slug: 'fracture', name: 'Bone Fracture Detection', desc: 'Binary classification + body region mapping', color: '#3B82F6', statsKey: 'fracture' },
    { slug: 'arthritis', name: 'Arthritis Grading', desc: '5-class KL score for knee radiographs', color: '#8B5CF6', statsKey: 'arthritis' },
    { slug: 'osteoporosis', name: 'Osteoporosis Screening', desc: 'DEXA-equivalent bone density analysis', color: '#EC4899', statsKey: 'osteoporosis' },
    { slug: 'tb', name: 'TB Screening', desc: 'Chest X-Ray tuberculosis detection', color: '#F59E0B', statsKey: 'tb' },
    { slug: 'lung-nodule', name: 'Lung Nodule Detection', desc: 'Malignancy risk assessment', color: '#10B981', statsKey: 'lung_nodule' },
    { slug: 'brain-tumor', name: 'Brain Tumor Classification', desc: '4-class MRI analysis (Glioma, Meningioma...)', color: '#EF4444', statsKey: 'brain_tumor' },
    { slug: 'brain-hemorrhage', name: 'Brain Hemorrhage', desc: 'Emergency ICH detection from CT scans', color: '#DC2626', statsKey: 'brain_hemorrhage' },
    { slug: 'bone-age', name: 'Bone Age Estimation', desc: 'Pediatric skeletal maturity regression', color: '#06B6D4', statsKey: 'bone_age' },
    { slug: 'retinopathy', name: 'Diabetic Retinopathy', desc: '5-class severity grading for fundus imaging', color: '#6366F1', statsKey: 'retinopathy' }
];

function renderModules(statsObj = {}) {
    const grid = document.getElementById('module-grid');
    grid.innerHTML = '';
    
    modules.forEach(mod => {
        const count = statsObj[mod.statsKey] || 0;
        const html = `
            <div class="bg-surface rounded-xl shadow-sm border border-slate-200 overflow-hidden relative module-card flex flex-col" style="border-left: 4px solid ${mod.color}" onclick="window.location.href='/scan/${mod.slug}'">
                <div class="p-6 flex-1">
                    <div class="flex items-center mb-4">
                        <div class="w-10 h-10 rounded-full flex items-center justify-center" style="background-color: ${mod.color}20">
                            <img src="/static/img/module-icons/${mod.slug}.svg" class="w-6 h-6">
                        </div>
                        <h3 class="ml-3 text-lg font-semibold text-slate-800 leading-tight">${mod.name}</h3>
                    </div>
                    <p class="text-sm text-slate-500 mb-4">${mod.desc}</p>
                </div>
                <div class="bg-slate-50 px-6 py-3 border-t border-slate-100 flex items-center justify-between">
                    <span class="text-xs font-medium text-slate-400">${count} scans today</span>
                    <button class="text-sm font-semibold transition-colors flex items-center" style="color: ${mod.color}">
                        Start Scan 
                        <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
                    </button>
                </div>
            </div>
        `;
        grid.insertAdjacentHTML('beforeend', html);
    });
}

function animateCounter(id, target) {
    const el = document.getElementById(id);
    if(!el) return;
    let current = 0;
    const inc = target / 20;
    const timer = setInterval(() => {
        current += inc;
        if(current >= target) {
            el.textContent = target;
            clearInterval(timer);
        } else {
            el.textContent = Math.ceil(current);
        }
    }, 30);
}

// Fetch stats on load
async function loadDashboardStats() {
    try {
        const res = await apiFetch('/admin/dashboard_stats');
        if (res && res.ok) {
            const data = await res.json();
            
            animateCounter('stat-patients', data.patients);
            animateCounter('stat-scans', data.total_scans);
            animateCounter('stat-alerts', data.alerts);
            animateCounter('stat-reports', data.reports);
            
            renderModules(data.module_counts);
        } else {
            renderModules();
        }
    } catch(err) {
        console.error("Stats Error:", err);
        renderModules();
    }
}

document.addEventListener('DOMContentLoaded', loadDashboardStats);
