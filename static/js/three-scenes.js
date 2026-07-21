// 1. Particle field (used on landing, login, dashboard background)
function createParticleField(containerId, particleCount = 1500, color = 0x14B8A6) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // Particles
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount * 3; i++) {
        positions[i] = (Math.random() - 0.5) * 50;
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({ color, size: 0.03, transparent: true, opacity: 0.6 });
    const particles = new THREE.Points(geometry, material);
    scene.add(particles);

    camera.position.z = 20;

    function animate() {
        requestAnimationFrame(animate);
        particles.rotation.y += 0.0003;
        particles.rotation.x += 0.0001;
        renderer.render(scene, camera);
    }
    animate();

    // Resize handler
    window.addEventListener('resize', () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });
}

// 2. Wireframe medical objects (landing page hero)
function createMedicalScene(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // Particles (same as field but denser)
    const geom = new THREE.BufferGeometry();
    const pos = new Float32Array(2000 * 3);
    for (let i = 0; i < 2000 * 3; i++) {
        pos[i] = (Math.random() - 0.5) * 50;
    }
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({ color: 0x14B8A6, size: 0.02, transparent: true, opacity: 0.5 });
    const particles = new THREE.Points(geom, mat);
    scene.add(particles);

    // Add wireframe shapes representing medical objects:
    const wireMat = new THREE.MeshBasicMaterial({ color: 0x22D3EE, wireframe: true, transparent: true, opacity: 0.3 });

    // Brain — IcosahedronGeometry (rough brain shape)
    const brainGeo = new THREE.IcosahedronGeometry(2.5, 2);
    const brain = new THREE.Mesh(brainGeo, wireMat);
    brain.position.set(-8, 2, -5);

    // Lung — Two SphereGeometry side by side
    const lungGeo = new THREE.SphereGeometry(1.8, 12, 12);
    const lungL = new THREE.Mesh(lungGeo, wireMat.clone());
    const lungR = new THREE.Mesh(lungGeo, wireMat.clone());
    lungL.position.set(6, -1, -8);
    lungR.position.set(8.5, -1, -8);

    // Spine — series of TorusGeometry stacked
    const spineGroup = new THREE.Group();
    for (let i = 0; i < 8; i++) {
        const vertebra = new THREE.Mesh(
            new THREE.TorusGeometry(0.6, 0.15, 8, 12),
            wireMat.clone()
        );
        vertebra.position.y = i * 0.7 - 2.5;
        vertebra.rotation.x = Math.PI / 2;
        spineGroup.add(vertebra);
    }
    spineGroup.position.set(0, 0, -10);

    // Eye — SphereGeometry with TorusGeometry iris
    const eyeball = new THREE.Mesh(new THREE.SphereGeometry(1.5, 16, 16), wireMat.clone());
    const iris = new THREE.Mesh(new THREE.TorusGeometry(0.7, 0.1, 8, 16), wireMat.clone());
    iris.position.z = 1.3;
    const eyeGroup = new THREE.Group();
    eyeGroup.add(eyeball, iris);
    eyeGroup.position.set(10, 3, -6);

    // Bone — CylinderGeometry with SphereGeometry ends
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 4, 8), wireMat.clone());
    const joint1 = new THREE.Mesh(new THREE.SphereGeometry(0.6, 8, 8), wireMat.clone());
    const joint2 = new THREE.Mesh(new THREE.SphereGeometry(0.6, 8, 8), wireMat.clone());
    joint1.position.y = 2;
    joint2.position.y = -2;
    const boneGroup = new THREE.Group();
    boneGroup.add(shaft, joint1, joint2);
    boneGroup.position.set(-6, -3, -7);
    boneGroup.rotation.z = Math.PI / 6;

    scene.add(brain, lungL, lungR, spineGroup, eyeGroup, boneGroup);

    camera.position.z = 20;

    // Animate: each object rotates slowly at different speeds
    let t = 0;
    function animate() {
        requestAnimationFrame(animate);
        t += 0.01;
        
        // Auto-orbit camera subtly
        camera.position.x = Math.sin(t * 0.1) * 2;
        camera.position.y = Math.cos(t * 0.1) * 2;
        camera.lookAt(scene.position);

        brain.rotation.y += 0.003;
        brain.rotation.x += 0.001;
        lungL.rotation.y += 0.002;
        lungR.rotation.y += 0.002;
        spineGroup.rotation.y += 0.001;
        eyeGroup.rotation.y += 0.004;
        boneGroup.rotation.y += 0.002;
        
        particles.rotation.y += 0.0002;

        renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });
}

function createParticleBackground(container, count = 800) {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count * 3; i++) {
        positions[i] = (Math.random() - 0.5) * 60;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({ color: 0x14B8A6, size: 0.03, transparent: true, opacity: 0.35, sizeAttenuation: true });
    const points = new THREE.Points(geo, mat);
    scene.add(points);

    camera.position.z = 25;

    function animate() {
        requestAnimationFrame(animate);
        points.rotation.y += 0.00015;
        points.rotation.x += 0.00005;
        renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });
}

function createParticleBackground(containerElement, particleCount = 800, color = 0x14B8A6) {
    if (!containerElement) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    containerElement.appendChild(renderer.domElement);

    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount * 3; i++) {
        positions[i] = (Math.random() - 0.5) * 50;
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({ color, size: 0.03, transparent: true, opacity: 0.6 });
    const particles = new THREE.Points(geometry, material);
    scene.add(particles);

    camera.position.z = 20;

    function animate() {
        requestAnimationFrame(animate);
        particles.rotation.y += 0.0003;
        particles.rotation.x += 0.0001;
        renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
}
