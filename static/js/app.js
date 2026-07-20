// Global Application State & Utilities
const API_BASE = window.location.origin;

// Auth
const getToken = () => sessionStorage.getItem('mediscan_token');
const getUser = () => JSON.parse(sessionStorage.getItem('mediscan_user') || 'null');
const setAuth = (token, user) => {
    sessionStorage.setItem('mediscan_token', token);
    sessionStorage.setItem('mediscan_user', JSON.stringify(user));
};
const clearAuth = () => {
    sessionStorage.removeItem('mediscan_token');
    sessionStorage.removeItem('mediscan_user');
};

// Authenticated fetch wrapper
async function apiFetch(url, options = {}) {
    const token = getToken();
    if (!token && !url.includes('/auth/')) {
        window.location.href = '/login';
        return;
    }
    const headers = { ...options.headers };
    if (token) headers['Authorization'] = `Bearer ${token}`;
    if (!(options.body instanceof FormData)) {
        headers['Content-Type'] = 'application/json';
    }
    
    try {
        const res = await fetch(API_BASE + url, { ...options, headers });
        if (res.status === 401 && !url.includes('/auth/')) {
            clearAuth();
            window.location.href = '/login';
            return;
        }
        return res;
    } catch (err) {
        console.error("API Fetch Error:", err);
        throw err;
    }
}

// Auth guards
function requireAuth() {
    if (!getToken()) {
        window.location.href = '/login';
    }
}
function requireAdmin() {
    const user = getUser();
    if (!user || user.role !== 'admin') {
        window.location.href = '/dashboard';
    }
}

// Load sidebar + navbar components
async function loadComponents() {
    try {
        const sidebarHtml = await fetch('/templates/components/sidebar.html').then(r => r.text());
        const navbarHtml = await fetch('/templates/components/navbar.html').then(r => r.text());
        
        const sidebarContainer = document.getElementById('sidebar-container');
        const navbarContainer = document.getElementById('navbar-container');
        
        if (sidebarContainer) sidebarContainer.innerHTML = sidebarHtml;
        if (navbarContainer) navbarContainer.innerHTML = navbarHtml;
        
        // Active nav link
        const path = window.location.pathname;
        document.querySelectorAll('.nav-link').forEach(link => {
            const href = link.getAttribute('href');
            if (path === href || (path.startsWith('/scan/') && href === '/dashboard') || (path.startsWith('/results/') && href === '/dashboard')) {
                link.classList.add('active');
            }
        });
        
        // Setup User Info
        const user = getUser();
        if (user) {
            const nameEl = document.getElementById('sidebar-user-name');
            const roleEl = document.getElementById('sidebar-user-role');
            const initEl = document.getElementById('sidebar-user-initials');
            
            if (nameEl) nameEl.textContent = user.full_name || 'Dr. Unknown';
            if (roleEl) roleEl.textContent = (user.role || 'Doctor').toUpperCase();
            if (initEl && user.full_name) {
                initEl.textContent = user.full_name.split(' ').map(n => n[0]).join('').substring(0, 2).toUpperCase();
            }
            
            const topNameEl = document.getElementById('nav-user-name');
            if (topNameEl) topNameEl.textContent = user.full_name;
        }
        
        // Hide Admin if not admin
        if (!user || user.role !== 'admin') {
            const adminLink = document.getElementById('nav-admin-link');
            if (adminLink) adminLink.style.display = 'none';
        }

        // Setup Logout
        const logoutBtn = document.getElementById('sidebar-logout-btn');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => {
                clearAuth();
                window.location.href = '/login';
            });
        }
        
        // Setup Sidebar Toggle
        const toggleBtn = document.getElementById('sidebar-toggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                const sidebar = document.getElementById('main-sidebar');
                const mainContent = document.getElementById('main-content-area');
                sidebar.classList.toggle('collapsed');
                if (window.innerWidth >= 1280) { // xl
                    if (sidebar.classList.contains('collapsed')) {
                        mainContent.style.marginLeft = '64px';
                        sidebar.style.width = '64px';
                    } else {
                        mainContent.style.marginLeft = '250px';
                        sidebar.style.width = '250px';
                    }
                }
            });
        }

    } catch(err) {
        console.error("Error loading components", err);
    }
}

// Toast System
function showToast(message, type = 'success') {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'fixed top-5 right-5 z-50 flex flex-col gap-2';
        document.body.appendChild(container);
    }
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type} p-4 rounded-lg shadow-lg flex items-center justify-between min-w-[300px] border-l-4 bg-white`;
    
    let icon = '';
    if (type === 'success') icon = '<svg class="w-5 h-5 text-emerald-600 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>';
    else if (type === 'error') icon = '<svg class="w-5 h-5 text-red-600 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>';
    else if (type === 'warning') icon = '<svg class="w-5 h-5 text-amber-600 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>';

    toast.innerHTML = `
        <div class="flex items-center text-slate-800 font-medium">${icon} ${message}</div>
        <button onclick="this.parentElement.remove()" class="text-slate-400 hover:text-slate-600">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
        </button>
    `;
    container.appendChild(toast);
    setTimeout(() => { if(toast.parentElement) toast.remove(); }, 5000);
}
