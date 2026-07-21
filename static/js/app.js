// === AUTH STATE ===
const API_BASE = '';
const getToken = () => sessionStorage.getItem('mediscan_token');
const getUser = () => JSON.parse(sessionStorage.getItem('mediscan_user') || 'null');

// === AUTHENTICATED FETCH ===
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
    const res = await fetch(API_BASE + url, { ...options, headers });
    if (res.status === 401) {
        sessionStorage.clear();
        window.location.href = '/login';
        return;
    }
    return res;
}

// === AUTH GUARDS ===
function requireAuth() { if (!getToken()) window.location.href = '/login'; }
function requireAdmin() {
    const user = getUser();
    if (!user || user.role !== 'admin') window.location.href = '/dashboard';
}

// === COMPONENT LOADER ===
async function loadComponents() {
    try {
        const [sidebarHtml, navbarHtml] = await Promise.all([
            fetch('/templates/components/sidebar.html').then(r => r.text()),
            fetch('/templates/components/navbar.html').then(r => r.text())
        ]);
        const sc = document.getElementById('sidebar-container');
        const nc = document.getElementById('navbar-container');
        if (sc) sc.innerHTML = sidebarHtml;
        if (nc) nc.innerHTML = navbarHtml;

        // Active nav link
        const path = window.location.pathname;
        document.querySelectorAll('.nav-link').forEach(link => {
            if (path.startsWith(link.getAttribute('href'))) {
                link.classList.add('active');
                link.querySelector('span')?.classList.add('text-teal-400');
                const leftBar = link.querySelector('.left-bar');
                if (leftBar) leftBar.classList.remove('opacity-0');
            }
        });

        // User info
        const user = getUser();
        if (user) {
            const nameEl = document.getElementById('user-name');
            const roleEl = document.getElementById('user-role');
            const initialsEl = document.getElementById('user-initials');
            if (nameEl) nameEl.textContent = user.full_name;
            if (roleEl) roleEl.textContent = user.role;
            if (initialsEl) initialsEl.textContent = user.full_name.split(' ').map(n => n[0]).join('').toUpperCase();
        }

        // Hide admin nav if not admin
        if (!user || user.role !== 'admin') {
            document.getElementById('admin-nav')?.classList.add('hidden');
        }
    } catch (e) {
        console.error('Failed to load components:', e);
    }
}

// === TOAST SYSTEM ===
function showToast(message, type = 'success') {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        document.body.appendChild(container);
    }
    const icons = { success: '✓', error: '✕', warning: '⚠' };
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `<span>${icons[type] || ''}</span> <span>${message}</span>
        <button onclick="this.parentElement.remove()" style="margin-left:auto;background:none;border:none;color:#94A3B8;cursor:pointer;font-size:18px;">×</button>`;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
}

// === 3D TILT EFFECT (reusable) ===
function initTiltCards() {
    document.querySelectorAll('.tilt-card').forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width - 0.5;
            const y = (e.clientY - rect.top) / rect.height - 0.5;
            card.style.transform = `perspective(1000px) rotateY(${x * 16}deg) rotateX(${-y * 16}deg) scale(1.02)`;
            card.style.transition = 'transform 0.1s ease';
        });
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateY(0) rotateX(0) scale(1)';
            card.style.transition = 'transform 0.5s ease';
        });
    });
}

// === GSAP PAGE ENTRANCE ===
function animatePageEntrance() {
    if (typeof gsap === 'undefined') return;
    gsap.from('.page-content', { opacity: 0, y: 30, duration: 0.6, ease: 'power2.out' });
    gsap.from('.stat-card', { opacity: 0, y: 20, duration: 0.5, stagger: 0.1, ease: 'power2.out', delay: 0.2 });
    gsap.from('.module-card', { opacity: 0, y: 20, duration: 0.5, stagger: 0.08, ease: 'power2.out', delay: 0.4 });
}

// === COUNTER ANIMATION ===
function animateCounter(el, target, duration = 1500) {
    let start = 0;
    const startTime = performance.now();
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        el.textContent = Math.floor(eased * target);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// === LOGOUT ===
function logout() {
    sessionStorage.clear();
    window.location.href = '/login';
}
