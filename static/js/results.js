document.addEventListener("DOMContentLoaded", async () => {
    requireAuth();
    await loadComponents();
    if (typeof createParticleBackground === 'function') {
        createParticleBackground(document.getElementById('particle-bg'), 800);
    }

    let currentScanData = null;
    let originalSrc = '';
    let heatmapSrc = '';

    // --- TAB SWITCHING ---
    const tabs = document.querySelectorAll('.img-tab');
    const imgEl = document.getElementById('result-image');
    
    window.switchImageTab = function(view) {
        tabs.forEach(t => {
            if (t.dataset.view === view) {
                t.classList.add('active');
                t.style.background = 'rgba(20,184,166,0.1)';
                t.style.borderColor = 'rgba(20,184,166,0.3)';
                t.style.color = '#14B8A6';
            } else {
                t.classList.remove('active');
                t.style.background = 'transparent';
                t.style.borderColor = 'rgba(255,255,255,0.06)';
                t.style.color = '#94A3B8';
            }
        });

        if (view === 'original') {
            imgEl.src = originalSrc;
        } else if (view === 'heatmap') {
            imgEl.src = heatmapSrc;
        } else {
            // side-by-side not fully implemented in new layout, fallback to heatmap
            imgEl.src = heatmapSrc;
        }
    }

    // --- LIGHTBOX ---
    window.openLightbox = function(src) {
        const lbImg = document.getElementById('lightbox-img');
        lbImg.src = src;
        const lb = document.getElementById('lightbox');
        lb.style.display = 'flex';
        gsap.from(lbImg, { scale: 0.9, opacity: 0, duration: 0.3, ease: 'power2.out' });
    }
    window.closeLightbox = function() {
        const lb = document.getElementById('lightbox');
        gsap.to(lb, { opacity: 0, duration: 0.2, onComplete: () => {
            lb.style.display = 'none';
            lb.style.opacity = '';
        }});
    }

    function animateConfidence(percentage) {
        const fill = document.getElementById('confidence-fill');
        const val = document.getElementById('confidence-value');
        let current = { p: 0 };
        gsap.to(current, {
            p: percentage,
            duration: 1.5,
            ease: "power2.out",
            delay: 0.5,
            onUpdate: () => {
                fill.style.width = current.p + '%';
                val.textContent = Math.round(current.p) + '%';
            }
        });
    }

    function setDiagnosisBadge(text, severity) {
        const badge = document.getElementById('diagnosis-badge');
        badge.textContent = text;
        
        let color, shadow;
        if (severity === 'critical') { color = '#F87171'; shadow = 'rgba(248,113,113,0.3)'; }
        else if (severity === 'warning') { color = '#FBBF24'; shadow = 'rgba(251,191,36,0.3)'; }
        else { color = '#34D399'; shadow = 'rgba(52,211,153,0.3)'; }

        badge.style.color = color;
        badge.style.background = color.replace(')', ', 0.1)').replace('rgb', 'rgba').replace('#', '');
        // hex fallback
        if (color.startsWith('#')) {
            const hex = color.replace('#', '');
            const r = parseInt(hex.substring(0,2),16);
            const g = parseInt(hex.substring(2,4),16);
            const b = parseInt(hex.substring(4,6),16);
            badge.style.background = `rgba(${r},${g},${b},0.15)`;
            badge.style.border = `1px solid rgba(${r},${g},${b},0.3)`;
        }

        gsap.to(badge, { opacity: 1, scale: 1, duration: 0.5, ease: 'back.out(1.5)', delay: 0.2 });
    }

    // --- FETCH DATA ---
    try {
        const res = await apiFetch(`/${window.MODULE_NAME}/scan/${window.SCAN_ID}`);
        if (!res.ok) throw new Error('Failed to load scan');
        
        const data = await res.json();
        currentScanData = data;
        
        // 1. Set Images
        const apiBase = window.API_BASE || '';
        originalSrc = apiBase + '/' + data.image_path;
        imgEl.src = originalSrc;
        
        if (data.heatmap_path) {
            heatmapSrc = apiBase + '/' + data.heatmap_path;
        } else {
            // Hide heatmap tab
            document.querySelector('[data-view="heatmap"]').classList.add('hidden');
        }

        // 2. Set Timestamps
        document.getElementById('scan-date').textContent = `Scan Date: ${new Date(data.timestamp).toLocaleString()}`;

        // 3. Parse Result & Fill UI
        let confidence = data.confidence ? Math.round(data.confidence * 100) : 0;
        let diagnosisText = data.result || 'Unknown';
        
        if (window.MODULE_NAME === 'bone_age') {
            const ageMonths = parseFloat(data.result);
            const years = Math.floor(ageMonths / 12);
            const months = Math.round(ageMonths % 12);
            diagnosisText = `${years}y ${months}m`;
            confidence = 95; 
        }

        let severity = 'success';
        let recText = 'No immediate action required based on AI analysis. Standard clinical correlation recommended.';

        const lRes = diagnosisText.toLowerCase();
        if (lRes.includes('normal') || lRes.includes('no') || lRes.includes('grade 0') || lRes.includes('negative')) {
            severity = 'success';
            recText = 'Findings are consistent with normal limits. Routine follow-up as clinically indicated.';
        } else if (lRes.includes('mild') || lRes.includes('grade 1') || lRes.includes('grade 2') || lRes.includes('early')) {
            severity = 'warning';
            recText = 'Early or mild signs detected. Recommend non-urgent specialist review and potential conservative management.';
        } else if (lRes.includes('severe') || lRes.includes('grade 3') || lRes.includes('grade 4') || lRes.includes('positive') || lRes.includes('glioma') || lRes.includes('tumor') || lRes.includes('hemorrhage')) {
            severity = 'critical';
            recText = 'URGENT: Significant abnormal findings detected. Immediate specialist consultation strongly recommended.';
        }

        setDiagnosisBadge(diagnosisText, severity);
        animateConfidence(confidence);
        document.getElementById('recommendation-text').textContent = recText;

    } catch (e) {
        showToast('Error loading scan results', 'error');
        console.error(e);
    }

    // --- OVERRIDE LOGIC ---
    const modal = document.getElementById('override-modal');
    const select = document.getElementById('override-diagnosis');

    const options = {
        'fracture': ['Fracture Detected', 'Normal'],
        'arthritis': ['Grade 0 (Normal)', 'Grade 1 (Doubtful)', 'Grade 2 (Mild)', 'Grade 3 (Moderate)', 'Grade 4 (Severe)'],
        'brain_tumor': ['Glioma', 'Meningioma', 'Pituitary Tumor', 'No Tumor'],
        'lung_nodule': ['Nodule Detected', 'Normal'],
        'tb': ['Positive', 'Normal']
    };

    window.showOverrideModal = function() {
        const modOptions = options[window.MODULE_NAME] || ['Positive', 'Negative', 'Normal', 'Abnormal'];
        select.innerHTML = '<option value="" disabled selected>Select new diagnosis</option>';
        modOptions.forEach(opt => {
            select.innerHTML += `<option value="${opt}">${opt}</option>`;
        });
        
        modal.style.display = 'flex';
        gsap.from(modal.querySelector('div'), { scale: 0.9, opacity: 0, duration: 0.3, ease: 'back.out(1.5)' });
    }

    window.closeOverrideModal = function() {
        modal.style.display = 'none';
    }

    window.submitOverride = async function() {
        const newDiag = select.value;
        const notes = document.getElementById('override-notes').value;
        
        if (!newDiag) return showToast('Please select a diagnosis', 'warning');

        try {
            const res = await apiFetch(`/${window.MODULE_NAME}/${window.SCAN_ID}/override`, {
                method: 'PATCH',
                body: JSON.stringify({
                    clinician_override: newDiag,
                    override_notes: notes
                })
            });

            if (res.ok) {
                showToast('Diagnosis updated successfully', 'success');
                setTimeout(() => window.location.reload(), 1000);
            } else {
                throw new Error('Failed to update');
            }
        } catch (e) {
            showToast(e.message, 'error');
        }
    }

    // --- REPORT DOWNLOAD ---
    window.downloadReport = async function() {
        try {
            showToast('Generating report...', 'info');
            const res = await apiFetch(`/${window.MODULE_NAME}/scan/${window.SCAN_ID}/report`);
            if (res.ok) {
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `MediScan_Report_${window.SCAN_ID}.pdf`;
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(url);
                showToast('Report downloaded', 'success');
            } else {
                throw new Error('Report generation failed');
            }
        } catch(e) {
            showToast(e.message, 'error');
        }
    }
});
