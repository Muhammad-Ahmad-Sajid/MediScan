const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 30);

const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x0B1426, 1);
document.getElementById('three-container').appendChild(renderer.domElement);

// 1. WIREFRAME BRAIN
const brainGeometry = new THREE.IcosahedronGeometry(14, 3);
const brainMaterial = new THREE.MeshBasicMaterial({
    color: 0x14B8A6,
    wireframe: true,
    transparent: true,
    opacity: 0.18
});
const brain = new THREE.Mesh(brainGeometry, brainMaterial);
brain.position.set(-8, 0, -10);
scene.add(brain);

// 2. DNA DOUBLE HELIX
function createDNAHelix() {
    const dnaGroup = new THREE.Group();
    const sphereGeo = new THREE.SphereGeometry(0.3, 8, 8);
    const mat1 = new THREE.MeshBasicMaterial({ color: 0x22D3EE, transparent: true, opacity: 0.6 });
    const mat2 = new THREE.MeshBasicMaterial({ color: 0x14B8A6, transparent: true, opacity: 0.6 });

    const points1 = [];
    const points2 = [];

    for (let i = 0; i < 60; i++) {
        const t = i * 0.15;
        const y = i * 0.45 - 13.5;

        const x1 = Math.cos(t) * 3.0;
        const z1 = Math.sin(t) * 3.0;
        const sphere1 = new THREE.Mesh(sphereGeo, mat1);
        sphere1.position.set(x1, y, z1);
        dnaGroup.add(sphere1);
        points1.push(new THREE.Vector3(x1, y, z1));

        const x2 = Math.cos(t + Math.PI) * 3.0;
        const z2 = Math.sin(t + Math.PI) * 3.0;
        const sphere2 = new THREE.Mesh(sphereGeo, mat2);
        sphere2.position.set(x2, y, z2);
        dnaGroup.add(sphere2);
        points2.push(new THREE.Vector3(x2, y, z2));

        if (i % 3 === 0) {
            const lineGeo = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(x1, y, z1),
                new THREE.Vector3(x2, y, z2)
            ]);
            const lineMat = new THREE.LineBasicMaterial({ color: 0x14B8A6, transparent: true, opacity: 0.25 });
            dnaGroup.add(new THREE.Line(lineGeo, lineMat));
        }
    }

    const strand1Geo = new THREE.BufferGeometry().setFromPoints(points1);
    const strand2Geo = new THREE.BufferGeometry().setFromPoints(points2);
    const strandMat = new THREE.LineBasicMaterial({ color: 0x22D3EE, transparent: true, opacity: 0.3 });
    dnaGroup.add(new THREE.Line(strand1Geo, strandMat));
    dnaGroup.add(new THREE.Line(strand2Geo, strandMat));

    return dnaGroup;
}

const dna = createDNAHelix();
dna.position.set(-10, 0, -3);
scene.add(dna);

// 3. ORBITING MEDICAL OBJECTS
const orbitObjects = [];
const wireMat = (color, opacity = 0.25) => new THREE.MeshBasicMaterial({
    color, wireframe: true, transparent: true, opacity
});

const lungGeo = new THREE.SphereGeometry(3.5, 12, 12);
const lung = new THREE.Mesh(lungGeo, wireMat(0x34D399, 0.2));
orbitObjects.push({ mesh: lung, radius: 14, speed: 0.003, yOffset: 4, phase: 0 });
scene.add(lung);

const eyeGroup = new THREE.Group();
const eyeball = new THREE.Mesh(new THREE.SphereGeometry(2.2, 12, 12), wireMat(0x818CF8, 0.2));
const iris = new THREE.Mesh(new THREE.TorusGeometry(1.0, 0.15, 8, 16), wireMat(0x818CF8, 0.35));
iris.position.z = 1.8;
eyeGroup.add(eyeball, iris);
orbitObjects.push({ mesh: eyeGroup, radius: 16, speed: 0.002, yOffset: -5, phase: Math.PI * 0.5 });
scene.add(eyeGroup);

const spineGroup = new THREE.Group();
for (let i = 0; i < 6; i++) {
    const vertebra = new THREE.Mesh(
        new THREE.TorusGeometry(1.0, 0.25, 8, 12),
        wireMat(0xF472B6, 0.2)
    );
    vertebra.position.y = i * 1.2 - 3;
    vertebra.rotation.x = Math.PI / 2;
    spineGroup.add(vertebra);
}
orbitObjects.push({ mesh: spineGroup, radius: 12, speed: 0.004, yOffset: -3, phase: Math.PI });
scene.add(spineGroup);

const boneGroup = new THREE.Group();
const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.4, 6, 8), wireMat(0x60A5FA, 0.2));
const j1 = new THREE.Mesh(new THREE.SphereGeometry(0.9, 8, 8), wireMat(0x60A5FA, 0.25));
const j2 = j1.clone();
j1.position.y = 3; j2.position.y = -3;
boneGroup.add(shaft, j1, j2);
boneGroup.rotation.z = Math.PI / 5;
orbitObjects.push({ mesh: boneGroup, radius: 18, speed: 0.0025, yOffset: 6, phase: Math.PI * 1.5 });
scene.add(boneGroup);

const heartGeo = new THREE.SphereGeometry(3, 10, 10);
heartGeo.scale(1, 1.2, 0.8);
const heart = new THREE.Mesh(heartGeo, wireMat(0xF87171, 0.2));
orbitObjects.push({ mesh: heart, radius: 15, speed: 0.0035, yOffset: 2, phase: Math.PI * 0.75 });
scene.add(heart);

// 4. PARTICLE FIELD
const particleCount = 3000;
const particleGeo = new THREE.BufferGeometry();
const positions = new Float32Array(particleCount * 3);
const sizes = new Float32Array(particleCount);

for (let i = 0; i < particleCount; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 80;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 80;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 80;
    sizes[i] = Math.random() * 0.04 + 0.01;
}

particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
const particleMat = new THREE.PointsMaterial({
    color: 0x14B8A6,
    size: 0.04,
    transparent: true,
    opacity: 0.5,
    sizeAttenuation: true
});
const particles = new THREE.Points(particleGeo, particleMat);
scene.add(particles);

// 5. AMBIENT LIGHTING
const light1 = new THREE.PointLight(0x14B8A6, 2, 50);
light1.position.set(0, 0, 10);
scene.add(light1);

const light2 = new THREE.PointLight(0x22D3EE, 1.5, 40);
light2.position.set(-15, 5, 5);
scene.add(light2);

// 6. MOUSE TRACKING
let mouseX = 0, mouseY = 0;
document.addEventListener('mousemove', (e) => {
    mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
    mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
});

// RESIZE HANDLER
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// ANIMATION LOOP
function animate() {
    requestAnimationFrame(animate);

    brain.rotation.y += 0.002;
    brain.rotation.x += 0.0005;

    dna.rotation.y += 0.008;

    const time = Date.now() * 0.001;
    orbitObjects.forEach(obj => {
        const angle = time * obj.speed * 10 + obj.phase;
        obj.mesh.position.x = Math.cos(angle) * obj.radius - 6;
        obj.mesh.position.z = Math.sin(angle) * obj.radius - 10;
        obj.mesh.position.y = obj.yOffset + Math.sin(time * 0.5 + obj.phase) * 2;
        obj.mesh.rotation.y += obj.speed;
        obj.mesh.rotation.x += obj.speed * 0.3;
    });

    particles.rotation.y += 0.0002;
    particles.rotation.x += 0.0001;

    camera.position.x += (mouseX * 3 - camera.position.x) * 0.02;
    camera.position.y += (-mouseY * 2 - camera.position.y) * 0.02;
    camera.lookAt(0, 0, -10);

    renderer.render(scene, camera);
}
animate();

// 7. LOGIN CARD 3D MOUSE EFFECT
const card = document.getElementById('login-card');
if(card) {
    card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width - 0.5;
        const y = (e.clientY - rect.top) / rect.height - 0.5;

        card.style.transform = `
            perspective(1200px)
            rotateY(${x * 20}deg)
            rotateX(${-y * 20}deg)
            translateZ(20px)
        `;

        card.style.boxShadow = `
            ${x * 40}px ${y * 40}px 80px rgba(20, 184, 166, 0.12),
            0 0 60px rgba(20, 184, 166, 0.06),
            0 25px 50px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.06)
        `;
    });

    card.addEventListener('mouseleave', () => {
        card.style.transform = 'perspective(1200px) rotateY(0) rotateX(0) translateZ(0)';
        card.style.transition = 'transform 0.6s ease, box-shadow 0.6s ease';
        card.style.boxShadow = `
            0 0 60px rgba(20, 184, 166, 0.08),
            0 0 120px rgba(20, 184, 166, 0.04),
            0 25px 50px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.06)
        `;
    });

    card.addEventListener('mouseenter', () => {
        card.style.transition = 'transform 0.1s ease';
    });
}

// 8. GSAP ENTRANCE ANIMATION
window.addEventListener('load', () => {
    const tl = gsap.timeline();
    gsap.set('#login-card', { opacity: 0, scale: 0.8, rotateX: 15, y: 40 });
    gsap.from('#three-container', { opacity: 0, duration: 1, ease: 'power2.out' });

    tl.to('#login-card', {
        opacity: 1, scale: 1, rotateX: 0, y: 0,
        duration: 1.2, ease: 'power3.out', delay: 0.5
    });
    tl.from('#login-card svg', { scale: 0, opacity: 0, duration: 0.5, ease: 'back.out(2)' }, '-=0.6');
    tl.from('#login-card h1, #login-card p', {
        opacity: 0, y: 15, stagger: 0.1, duration: 0.4, ease: 'power2.out'
    }, '-=0.3');

    gsap.from('#left-branding h1', { opacity: 0, x: -40, duration: 1, ease: 'power2.out', delay: 0.3 });
    gsap.from('#left-branding p', { opacity: 0, x: -30, duration: 0.8, ease: 'power2.out', delay: 0.6 });
    gsap.from('#left-branding span', { opacity: 0, y: 10, stagger: 0.15, duration: 0.5, ease: 'power2.out', delay: 0.9 });
});

// 9. API LOGIC & FORMS
function getRegisterHTML() {
    return `
        <div class="flex justify-center mb-6">
            <div class="relative">
                <svg width="64" height="64" viewBox="0 0 64 64" class="drop-shadow-[0_0_20px_rgba(20,184,166,0.5)]">
                    <circle cx="32" cy="32" r="28" fill="none" stroke="rgba(20,184,166,0.3)" stroke-width="1.5"/>
                    <polyline points="12,32 22,32 26,20 30,44 34,28 38,36 42,32 52,32"
                              fill="none" stroke="#14B8A6" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <div class="absolute inset-0 rounded-full animate-ping opacity-20" style="background: radial-gradient(circle, rgba(20,184,166,0.4), transparent 70%);"></div>
            </div>
        </div>
        <h1 class="text-2xl font-bold text-center mb-1" style="color: #F1F5F9; text-shadow: 0 0 30px rgba(20,184,166,0.2);">
            Create Account
        </h1>
        <p class="text-center text-sm mb-6" style="color: #64748B;">
            Join MediScan AI Platform
        </p>
        <div class="mb-4">
            <label class="block text-xs font-semibold mb-2 tracking-wider" style="color: #94A3B8;">FULL NAME</label>
            <input type="text" id="reg-name" placeholder="Dr. Full Name"
                class="w-full px-4 py-3 rounded-xl text-sm outline-none placeholder-slate-400"
                style="background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #F1F5F9; transition: border-color 0.3s, box-shadow 0.3s;"
                onfocus="this.style.borderColor='rgba(20,184,166,0.5)'; this.style.boxShadow='0 0 20px rgba(20,184,166,0.1)'"
                onblur="this.style.borderColor='rgba(255,255,255,0.08)'; this.style.boxShadow='none'">
        </div>
        <div class="mb-4">
            <label class="block text-xs font-semibold mb-2 tracking-wider" style="color: #94A3B8;">EMAIL</label>
            <input type="email" id="reg-email" placeholder="doctor@hospital.com"
                class="w-full px-4 py-3 rounded-xl text-sm outline-none placeholder-slate-400"
                style="background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #F1F5F9; transition: border-color 0.3s, box-shadow 0.3s;"
                onfocus="this.style.borderColor='rgba(20,184,166,0.5)'; this.style.boxShadow='0 0 20px rgba(20,184,166,0.1)'"
                onblur="this.style.borderColor='rgba(255,255,255,0.08)'; this.style.boxShadow='none'">
        </div>
        <div class="grid grid-cols-2 gap-3 mb-6">
            <div>
                <label class="block text-xs font-semibold mb-2 tracking-wider" style="color: #94A3B8;">PASSWORD</label>
                <input type="password" id="reg-password" placeholder="Create a password"
                    class="w-full px-4 py-3 rounded-xl text-sm outline-none transition-all duration-300"
                    style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); color: #F1F5F9;"
                    onfocus="this.style.borderColor='rgba(20,184,166,0.5)'; this.style.boxShadow='0 0 20px rgba(20,184,166,0.1)'"
                    onblur="this.style.borderColor='rgba(255,255,255,0.08)'; this.style.boxShadow='none'">
            </div>
            <div>
                <label class="block text-xs font-semibold mb-2 tracking-wider" style="color: #94A3B8;">ROLE</label>
                <select id="reg-role"
                    class="w-full px-4 py-3 rounded-xl text-sm outline-none transition-all duration-300"
                    style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); color: #F1F5F9;"
                    onfocus="this.style.borderColor='rgba(20,184,166,0.5)'; this.style.boxShadow='0 0 20px rgba(20,184,166,0.1)'"
                    onblur="this.style.borderColor='rgba(255,255,255,0.08)'; this.style.boxShadow='none'">
                    <option value="doctor" style="background:#0B1426">Doctor</option>
                    <option value="admin" style="background:#0B1426">Admin</option>
                </select>
            </div>
        </div>
        <button id="register-btn" onclick="handleRegister()"
            class="w-full py-3.5 rounded-xl font-semibold text-sm"
            style="background: linear-gradient(135deg, #0F766E, #22D3EE); color: white; box-shadow: 0 0 30px rgba(20,184,166,0.2); transition: transform 0.3s, box-shadow 0.3s;"
            onmouseover="this.style.boxShadow='0 0 50px rgba(20,184,166,0.35)'; this.style.transform='translateY(-1px)'"
            onmouseout="this.style.boxShadow='0 0 30px rgba(20,184,166,0.2)'; this.style.transform='translateY(0)'">
            Create Account
        </button>
        <div id="register-error" class="hidden mt-4 text-center text-sm" style="color: #F87171;"></div>
        <p class="text-center mt-6 text-sm" style="color: #64748B;">
            Already have an account?
            <a href="#" onclick="showLogin()" style="color: #14B8A6; font-weight: 600;" class="hover:underline">Sign In</a>
        </p>
    `;
}

function getLoginHTML() {
    return `
        <div class="flex justify-center mb-6">
            <div class="relative">
                <svg width="64" height="64" viewBox="0 0 64 64" class="drop-shadow-[0_0_20px_rgba(20,184,166,0.5)]">
                    <circle cx="32" cy="32" r="28" fill="none" stroke="rgba(20,184,166,0.3)" stroke-width="1.5"/>
                    <polyline points="12,32 22,32 26,20 30,44 34,28 38,36 42,32 52,32"
                              fill="none" stroke="#14B8A6" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <div class="absolute inset-0 rounded-full animate-ping opacity-20" style="background: radial-gradient(circle, rgba(20,184,166,0.4), transparent 70%);"></div>
            </div>
        </div>
        <h1 class="text-2xl font-bold text-center mb-1" style="color: #F1F5F9; text-shadow: 0 0 30px rgba(20,184,166,0.2);">
            Welcome Back
        </h1>
        <p class="text-center text-sm mb-8" style="color: #64748B;">
            Sign in to MediScan AI
        </p>
        <div class="mb-5">
            <label class="block text-xs font-semibold mb-2 tracking-wider" style="color: #94A3B8;">EMAIL</label>
            <div class="relative">
                <input type="email" id="login-email" placeholder="Enter your email"
                    class="w-full px-4 py-3 rounded-xl text-sm outline-none transition-all duration-300"
                    style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); color: #F1F5F9;"
                    onfocus="this.style.borderColor='rgba(20,184,166,0.5)'; this.style.boxShadow='0 0 20px rgba(20,184,166,0.1)'"
                    onblur="this.style.borderColor='rgba(255,255,255,0.08)'; this.style.boxShadow='none'">
            </div>
        </div>
        <div class="mb-6">
            <label class="block text-xs font-semibold mb-2 tracking-wider" style="color: #94A3B8;">PASSWORD</label>
            <div class="relative">
                <input type="password" id="login-password" placeholder="Enter your password"
                    class="w-full px-4 py-3 rounded-xl text-sm outline-none transition-all duration-300"
                    style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); color: #F1F5F9;"
                    onfocus="this.style.borderColor='rgba(20,184,166,0.5)'; this.style.boxShadow='0 0 20px rgba(20,184,166,0.1)'"
                    onblur="this.style.borderColor='rgba(255,255,255,0.08)'; this.style.boxShadow='none'">
                <button type="button" onclick="togglePassword()" class="absolute right-3 top-1/2 -translate-y-1/2"
                    style="color: #64748B; background:none; border:none; cursor:pointer;">
                    <svg id="eye-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
                        <circle cx="12" cy="12" r="3"/>
                    </svg>
                </button>
            </div>
        </div>
        <button id="login-btn" onclick="handleLogin()"
            class="w-full py-3.5 rounded-xl font-semibold text-sm"
            style="background: linear-gradient(135deg, #0F766E, #22D3EE); color: white; box-shadow: 0 0 30px rgba(20,184,166,0.2); transition: transform 0.3s, box-shadow 0.3s;"
            onmouseover="this.style.boxShadow='0 0 50px rgba(20,184,166,0.35)'; this.style.transform='translateY(-1px)'"
            onmouseout="this.style.boxShadow='0 0 30px rgba(20,184,166,0.2)'; this.style.transform='translateY(0)'">
            Sign In
        </button>
        <div id="login-error" class="hidden mt-4 text-center text-sm" style="color: #F87171;"></div>
        <p class="text-center mt-6 text-sm" style="color: #64748B;">
            Don't have an account?
            <a href="#" onclick="showRegister()" style="color: #14B8A6; font-weight: 600;" class="hover:underline">Register</a>
        </p>
    `;
}

function showRegister() {
    const card = document.getElementById('login-card');
    gsap.to(card, {
        rotateY: 90, opacity: 0, duration: 0.3, ease: 'power2.in',
        onComplete: () => {
            card.querySelector('[style*="z-index:2"]').innerHTML = getRegisterHTML();
            gsap.fromTo(card,
                { rotateY: -90, opacity: 0 },
                { rotateY: 0, opacity: 1, duration: 0.4, ease: 'power2.out' }
            );
        }
    });
}

function showLogin() {
    const card = document.getElementById('login-card');
    gsap.to(card, {
        rotateY: -90, opacity: 0, duration: 0.3, ease: 'power2.in',
        onComplete: () => {
            card.querySelector('[style*="z-index:2"]').innerHTML = getLoginHTML();
            gsap.fromTo(card,
                { rotateY: 90, opacity: 0 },
                { rotateY: 0, opacity: 1, duration: 0.4, ease: 'power2.out' }
            );
        }
    });
}

function togglePassword() {
    const input = document.getElementById('login-password') || document.getElementById('reg-password');
    if (input) {
        input.type = input.type === 'password' ? 'text' : 'password';
    }
}

async function handleLogin() {
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;
    const errorEl = document.getElementById('login-error');
    const btn = document.getElementById('login-btn');

    if (!email || !password) {
        errorEl.textContent = 'Please fill in all fields';
        errorEl.classList.remove('hidden');
        return;
    }

    btn.textContent = 'Signing in...';
    btn.disabled = true;

    try {
        const formData = new URLSearchParams();
        formData.append('username', email);
        formData.append('password', password);

        const res = await fetch('/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: formData
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Invalid credentials');
        }

        const data = await res.json();
        sessionStorage.setItem('mediscan_token', data.access_token);

        const userRes = await fetch('/auth/me', {
            headers: { 'Authorization': `Bearer ${data.access_token}` }
        });
        const user = await userRes.json();
        sessionStorage.setItem('mediscan_user', JSON.stringify(user));

        if (typeof showToast === 'function') showToast('Login successful!', 'success');

        gsap.to('#login-card', {
            scale: 0.9, opacity: 0, rotateX: -10, y: -30,
            duration: 0.5, ease: 'power2.in',
            onComplete: () => { window.location.href = '/dashboard'; }
        });

    } catch (e) {
        errorEl.textContent = e.message;
        errorEl.classList.remove('hidden');
        btn.textContent = 'Sign In';
        btn.disabled = false;

        gsap.to('#login-card', {
            x: [-10, 10, -8, 8, -4, 4, 0],
            duration: 0.5,
            ease: 'power2.out'
        });
    }
}

async function handleRegister() {
    const name = document.getElementById('reg-name').value;
    const email = document.getElementById('reg-email').value;
    const password = document.getElementById('reg-password').value;
    const role = document.getElementById('reg-role').value;
    const errorEl = document.getElementById('register-error');
    const btn = document.getElementById('register-btn');

    if (!name || !email || !password) {
        errorEl.textContent = 'Please fill in all fields';
        errorEl.classList.remove('hidden');
        return;
    }

    btn.textContent = 'Creating Account...';
    btn.disabled = true;

    try {
        const res = await fetch('/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ full_name: name, email, password, role })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Registration failed');
        }

        if (typeof showToast === 'function') showToast('Registration successful! Please sign in.', 'success');
        
        showLogin();
        
        setTimeout(() => {
            const loginEmailInput = document.getElementById('login-email');
            if (loginEmailInput) loginEmailInput.value = email;
        }, 500);

    } catch (e) {
        errorEl.textContent = e.message;
        errorEl.classList.remove('hidden');
        btn.textContent = 'Create Account';
        btn.disabled = false;
    }
}
