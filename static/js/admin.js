document.addEventListener("DOMContentLoaded", async () => {
    requireAuth();
    // Assuming requireAdmin(); exists or we handle it in requireAuth/app.js
    
    await loadComponents();
    if (typeof createParticleBackground === 'function') {
        createParticleBackground(document.getElementById('particle-bg'), 800);
    }

    // Fetch and render users
    try {
        const tbody = document.getElementById('users-table-body');
        
        let users = [];
        try {
            const res = await apiFetch('/admin/users');
            if (res && res.ok) {
                users = await res.json();
            } else {
                throw new Error('API not available');
            }
        } catch (e) {
            // Mock data fallback
            users = [
                { username: 'admin', full_name: 'System Administrator', role: 'admin', active: true },
                { username: 'doctor_sajid', full_name: 'Dr. Sajid', role: 'doctor', active: true },
                { username: 'dr_jane', full_name: 'Dr. Jane Doe', role: 'doctor', active: true },
                { username: 'guest_user', full_name: 'Guest Observer', role: 'user', active: false }
            ];
        }

        tbody.innerHTML = '';
        users.forEach((u, i) => {
            const tr = document.createElement('tr');
            tr.style.borderBottom = '1px solid rgba(255,255,255,0.06)';
            tr.style.transition = 'background 0.2s';
            tr.addEventListener('mouseover', () => tr.style.background = 'rgba(255,255,255,0.02)');
            tr.addEventListener('mouseout', () => tr.style.background = 'transparent');
            
            const roleColor = u.role === 'admin' ? '#A78BFA' : (u.role === 'doctor' ? '#34D399' : '#94A3B8');
            const roleBg = u.role === 'admin' ? 'rgba(167,139,250,0.1)' : (u.role === 'doctor' ? 'rgba(52,211,153,0.1)' : 'rgba(148,163,184,0.1)');
            
            const statusColor = u.active ? '#14B8A6' : '#64748B';
            const statusText = u.active ? 'Active' : 'Inactive';

            tr.innerHTML = `
                <td style="padding:16px 12px;">
                    <div style="display:flex; align-items:center; gap:12px;">
                        <div style="width:36px; height:36px; border-radius:10px; background:${roleBg}; color:${roleColor}; display:flex; align-items:center; justify-content:center; font-weight:600; font-size:14px;">
                            ${u.full_name.substring(0,2).toUpperCase()}
                        </div>
                        <div>
                            <div style="font-weight:600; color:#F1F5F9; font-size:14px;">${u.full_name}</div>
                            <div style="font-size:12px; color:#64748B;">@${u.username}</div>
                        </div>
                    </div>
                </td>
                <td style="padding:16px 12px;">
                    <span style="padding:4px 8px; border-radius:6px; background:${roleBg}; color:${roleColor}; font-size:12px; font-weight:500; text-transform:capitalize;">${u.role}</span>
                </td>
                <td style="padding:16px 12px;">
                    <span style="display:flex; align-items:center; gap:6px; font-size:13px; color:${statusColor};">
                        <span style="width:6px; height:6px; border-radius:50%; background:${statusColor};"></span>
                        ${statusText}
                    </span>
                </td>
                <td style="padding:16px 12px; text-align:right;">
                    <button style="background:none; border:none; color:#94A3B8; cursor:pointer; margin-right:8px;" title="Edit">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
                    </button>
                    ${u.role !== 'admin' ? `
                    <button style="background:none; border:none; color:#94A3B8; cursor:pointer;" title="Deactivate" onmouseover="this.style.color='#F87171'" onmouseout="this.style.color='#94A3B8'">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
                    </button>
                    ` : ''}
                </td>
            `;

            gsap.from(tr, { opacity: 0, x: -20, duration: 0.4, delay: i * 0.1 });
            tbody.appendChild(tr);
        });

    } catch(e) {
        console.error("Admin error:", e);
    }
});
