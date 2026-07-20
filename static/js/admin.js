requireAuth();
requireAdmin(); // extra guard
loadComponents();

let allUsers = [];
let overrides = [];

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const roleFilter = document.getElementById('user-role-filter');
        if (roleFilter) {
            roleFilter.addEventListener('change', renderUsers);
        }

        await Promise.all([
            loadUsers(),
            loadOverrides(),
            loadStats()
        ]);
    } catch(err) {
        console.error("Admin init error:", err);
    }
});

async function loadStats() {
    try {
        const res = await apiFetch('/admin/dashboard_stats');
        if(res && res.ok) {
            const data = await res.json();
            document.getElementById('stat-patients').textContent = data.patients || 0;
            document.getElementById('stat-scans').textContent = data.total_scans || 0;
        } else {
            document.getElementById('stat-patients').textContent = 'Error';
            document.getElementById('stat-scans').textContent = 'Error';
        }
    } catch(err) {
        document.getElementById('stat-patients').textContent = 'Error';
        document.getElementById('stat-scans').textContent = 'Error';
    }
}

async function loadUsers() {
    try {
        const res = await apiFetch('/admin/users');
        if(res && res.ok) {
            allUsers = await res.json();
            document.getElementById('stat-users').textContent = allUsers.length;
            renderUsers();
        }
    } catch(err) {
        showToast('Failed to load users', 'error');
    }
}

function renderUsers() {
    const tbody = document.getElementById('user-table-body');
    tbody.innerHTML = '';
    
    const filterRole = document.getElementById('user-role-filter').value;
    const filtered = allUsers.filter(u => filterRole === 'all' || u.role === filterRole);
    
    filtered.forEach(u => {
        const statusBadge = u.is_active !== false 
            ? '<span class="bg-emerald-100 text-emerald-700 px-2 py-1 rounded-full text-xs font-bold">Active</span>'
            : '<span class="bg-red-100 text-red-700 px-2 py-1 rounded-full text-xs font-bold">Inactive</span>';
            
        const roleBadge = u.role === 'admin'
            ? '<span class="bg-indigo-100 text-indigo-700 px-2 py-1 rounded text-xs font-bold">Admin</span>'
            : '<span class="bg-slate-100 text-slate-700 px-2 py-1 rounded text-xs font-bold">Doctor</span>';

        tbody.insertAdjacentHTML('beforeend', `
            <tr class="hover:bg-slate-50 transition-colors">
                <td class="px-6 py-4">
                    <div class="flex items-center">
                        <div class="w-8 h-8 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center font-bold text-xs mr-3">
                            ${u.full_name.substring(0,2).toUpperCase()}
                        </div>
                        <span class="font-medium text-slate-800">${u.full_name}</span>
                    </div>
                </td>
                <td class="px-6 py-4 text-slate-500">${u.email}</td>
                <td class="px-6 py-4">${roleBadge}</td>
                <td class="px-6 py-4">${statusBadge}</td>
            </tr>
        `);
    });
}

async function loadOverrides() {
    try {
        const res = await apiFetch('/admin/overrides');
        if(res && res.ok) {
            overrides = await res.json();
            document.getElementById('stat-overrides').textContent = overrides.length;
            renderOverrides();
        }
    } catch(err) {
        showToast('Failed to load overrides', 'error');
    }
}

function renderOverrides() {
    const tbody = document.getElementById('override-table-body');
    tbody.innerHTML = '';
    
    if(overrides.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="px-6 py-8 text-center text-slate-500">No clinician overrides found.</td></tr>';
        return;
    }
    
    overrides.forEach(ov => {
        tbody.insertAdjacentHTML('beforeend', `
            <tr class="hover:bg-slate-50 transition-colors">
                <td class="px-6 py-4 whitespace-nowrap text-xs text-slate-500">${ov.date}</td>
                <td class="px-6 py-4 font-medium text-slate-800">${ov.doctor}</td>
                <td class="px-6 py-4"><span class="bg-slate-100 text-slate-700 px-2 py-1 rounded text-xs font-bold">${ov.module}</span></td>
                <td class="px-6 py-4 text-slate-500">${ov.patient}</td>
                <td class="px-6 py-4"><span class="bg-amber-100 text-amber-700 px-2 py-1 rounded-full text-xs font-bold border border-amber-200">${ov.override}</span></td>
                <td class="px-6 py-4 text-slate-500 text-xs max-w-xs truncate" title="${ov.notes}">${ov.notes || '-'}</td>
            </tr>
        `);
    });
}
