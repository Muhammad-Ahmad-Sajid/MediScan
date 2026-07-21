document.addEventListener('DOMContentLoaded', async () => {
    requireAuth();
    await loadComponents();
    createParticleBackground(document.getElementById('particle-bg'), 800);

    const titleEl = document.getElementById('page-title');
    const breadEl = document.getElementById('page-breadcrumb');
    if (titleEl) titleEl.textContent = 'Dashboard';
    if (breadEl) breadEl.textContent = 'MediScan AI / Dashboard';

    // Init 3D tilt on module cards
    if (typeof initTiltCards === 'function') initTiltCards();

    // Mouse glow effect on cards
    document.querySelectorAll('.module-card').forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = ((e.clientX - rect.left) / rect.width) * 100;
            const y = ((e.clientY - rect.top) / rect.height) * 100;
            const glow = card.querySelector('.hover-glow');
            if (glow) {
                glow.style.setProperty('--mouse-x', x + '%');
                glow.style.setProperty('--mouse-y', y + '%');
                glow.style.opacity = '1';
            }
        });
        card.addEventListener('mouseleave', (e) => {
            const glow = card.querySelector('.hover-glow');
            if (glow) glow.style.opacity = '0';
        });
    });

    // GSAP entrance
    gsap.from('.stat-card', { opacity: 0, y: 25, duration: 0.6, stagger: 0.1, ease: 'power2.out' });
    gsap.from('.module-card', { opacity: 0, y: 30, duration: 0.5, stagger: 0.06, ease: 'power2.out', delay: 0.3 });

    // Helper: animate counter
    function animateCounter(el, targetValue, duration) {
        let current = { val: 0 };
        gsap.to(current, {
            val: targetValue,
            duration: duration / 1000,
            ease: "power2.out",
            onUpdate: () => { el.textContent = Math.floor(current.val); }
        });
    }

    // Fetch real stats from API
    try {
        const res = await apiFetch('/admin/dashboard_stats');
        if (res && res.ok) {
            const data = await res.json();
            const statEls = document.querySelectorAll('.stat-number');
            if (statEls.length === 4) {
                animateCounter(statEls[0], data.total_scans || 0, 1800);
                animateCounter(statEls[1], data.total_patients || 0, 1800);
                animateCounter(statEls[2], data.critical_alerts || 0, 1800);
                animateCounter(statEls[3], data.reports_generated || 0, 1800);
            }
        } else {
            // Fallback to data-target
            document.querySelectorAll('.stat-number').forEach(el => {
                animateCounter(el, parseInt(el.dataset.target), 1800);
            });
        }
    } catch (e) { 
        console.error("Failed to fetch dashboard stats", e);
        // Fallback to data-target
        document.querySelectorAll('.stat-number').forEach(el => {
            animateCounter(el, parseInt(el.dataset.target), 1800);
        });
    }
});
