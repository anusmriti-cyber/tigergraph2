/* ============================================================
   Splash → Dashboard transition & extra micro-animations
   ============================================================ */

// --- Typed tagline effect ---
(function () {
    const tagline = document.getElementById('tagline');
    if (!tagline) return;

    const fullText = tagline.textContent;
    tagline.textContent = '';

    // Wait for the slide-up animation to finish (1.3s delay + 0.8s duration)
    setTimeout(() => {
        tagline.style.opacity = '1';
        tagline.style.transform = 'translateY(0)';
        // Override animation so we can type manually
        tagline.style.animation = 'none';

        let i = 0;
        const speed = 30; // ms per character

        function typeChar() {
            if (i < fullText.length) {
                tagline.textContent += fullText.charAt(i);
                i++;
                setTimeout(typeChar, speed);
            }
        }
        typeChar();
    }, 2100);
})();


// --- Logo tilt on mouse move (parallax) ---
(function () {
    const orbitContainer = document.getElementById('orbitContainer');
    if (!orbitContainer) return;

    document.addEventListener('mousemove', (e) => {
        const cx = window.innerWidth / 2;
        const cy = window.innerHeight / 2;
        const dx = (e.clientX - cx) / cx;
        const dy = (e.clientY - cy) / cy;
        const tiltX = dy * 8;   // degrees
        const tiltY = -dx * 8;

        orbitContainer.style.transform =
            `perspective(800px) rotateX(${tiltX}deg) rotateY(${tiltY}deg)`;
    });
})();


// --- "Enter Dashboard" transition ---
function enterDashboard() {
    const splash = document.getElementById('splashScreen');
    const dashboard = document.getElementById('dashboardScreen');

    // Fade out splash
    splash.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    splash.style.opacity = '0';
    splash.style.transform = 'scale(1.04)';

    setTimeout(() => {
        splash.classList.add('hidden');
        dashboard.classList.remove('hidden');
        document.body.style.overflow = 'auto';
    }, 600);
}


// --- Badge hover shimmer on intersection ---
(function () {
    const badges = document.querySelectorAll('.badge');
    badges.forEach((badge) => {
        badge.addEventListener('mouseenter', () => {
            badge.style.borderColor = 'rgba(0, 240, 255, 0.25)';
            badge.style.color = '#f0f0f5';
        });
        badge.addEventListener('mouseleave', () => {
            badge.style.borderColor = 'rgba(255,255,255,0.08)';
            badge.style.color = 'rgba(240,240,245,0.55)';
        });
    });
})();


// --- Scanline overlay for cyberpunk feel ---
(function () {
    const scanline = document.createElement('div');
    scanline.style.cssText = `
        position: fixed;
        inset: 0;
        z-index: 9999;
        pointer-events: none;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0,0,0,0.03) 2px,
            rgba(0,0,0,0.03) 4px
        );
    `;
    document.body.appendChild(scanline);
})();

// --- Hero Image Parallax (Cursor tracking) ---
(function () {
    const heroWrapper = document.querySelector('.hero-image-wrapper');
    if (!heroWrapper) return;
    
    // Set initial custom properties for the CSS animation
    heroWrapper.style.setProperty('--rotX', '10deg');
    heroWrapper.style.setProperty('--rotY', '-15deg');

    document.addEventListener('mousemove', (e) => {
        // Calculate mouse position relative to center of screen
        const cx = window.innerWidth / 2;
        const cy = window.innerHeight / 2;
        
        // Intensity of rotation (lower = less rotation)
        const intensity = 30;
        
        // Calculate rotation delta based on mouse pos
        const rotY = (e.clientX - cx) / intensity; 
        const rotX = -(e.clientY - cy) / intensity;
        
        // Base slanted position is X:10deg, Y:-15deg
        // Add the mouse delta
        const finalRotX = 10 + rotX;
        const finalRotY = -15 + rotY;
        
        heroWrapper.style.setProperty('--rotX', `${finalRotX}deg`);
        heroWrapper.style.setProperty('--rotY', `${finalRotY}deg`);
    });
})();
