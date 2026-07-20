// Redirect if already logged in
if (getToken()) {
    window.location.href = '/dashboard';
}

function togglePwd(id) {
    const input = document.getElementById(id);
    if (input.type === 'password') {
        input.type = 'text';
    } else {
        input.type = 'password';
    }
}

function toggleAuthView(view) {
    const loginView = document.getElementById('login-view');
    const registerView = document.getElementById('register-view');
    
    if (view === 'register') {
        loginView.style.transform = 'translateX(-100%)';
        registerView.style.transform = 'translateX(-100%)'; // Move it into view
    } else {
        loginView.style.transform = 'translateX(0)';
        registerView.style.transform = 'translateX(0)'; // Move it out of view
    }
}

document.getElementById('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = e.target.querySelector('button[type="submit"]');
    const origText = btn.textContent;
    btn.textContent = 'Signing in...';
    btn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('username', document.getElementById('login-email').value);
        formData.append('password', document.getElementById('login-password').value);

        const res = await fetch(API_BASE + '/auth/login', {
            method: 'POST',
            body: formData
        });

        const data = await res.json();
        
        if (res.ok && data.access_token) {
            setAuth(data.access_token, data.user);
            window.location.href = '/dashboard';
        } else {
            showToast(data.detail || 'Invalid credentials', 'error');
        }
    } catch (err) {
        showToast('Connection error', 'error');
    } finally {
        btn.textContent = origText;
        btn.disabled = false;
    }
});

document.getElementById('register-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = e.target.querySelector('button[type="submit"]');
    const origText = btn.textContent;
    btn.textContent = 'Creating...';
    btn.disabled = true;

    try {
        const payload = {
            email: document.getElementById('reg-email').value,
            password: document.getElementById('reg-password').value,
            full_name: document.getElementById('reg-name').value,
            role: document.getElementById('reg-role').value
        };

        const res = await fetch(API_BASE + '/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        
        if (res.ok) {
            showToast('Registration successful! Please sign in.', 'success');
            toggleAuthView('login');
            document.getElementById('login-email').value = payload.email;
        } else {
            showToast(data.detail || 'Registration failed', 'error');
        }
    } catch (err) {
        showToast('Connection error', 'error');
    } finally {
        btn.textContent = origText;
        btn.disabled = false;
    }
});
