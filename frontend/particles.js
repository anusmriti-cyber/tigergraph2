/* ============================================================
   Particle Network Background
   ============================================================ */
(function () {
    const canvas = document.getElementById('particleCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let W, H;
    const particles = [];
    const PARTICLE_COUNT = 80;
    const CONNECTION_DIST = 140;
    const MOUSE = { x: null, y: null, radius: 160 };

    function resize() {
        W = canvas.width = window.innerWidth;
        H = canvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    window.addEventListener('mousemove', (e) => {
        MOUSE.x = e.clientX;
        MOUSE.y = e.clientY;
    });
    window.addEventListener('mouseleave', () => {
        MOUSE.x = null;
        MOUSE.y = null;
    });

    class Particle {
        constructor() {
            this.x = Math.random() * W;
            this.y = Math.random() * H;
            this.vx = (Math.random() - 0.5) * 0.4;
            this.vy = (Math.random() - 0.5) * 0.4;
            this.r = Math.random() * 1.8 + 0.6;

            // pick from cyan / purple / magenta
            const palette = [
                { r: 0, g: 240, b: 255 },
                { r: 168, g: 85, b: 247 },
                { r: 224, g: 64, b: 251 },
                { r: 59, g: 130, b: 246 }
            ];
            this.color = palette[Math.floor(Math.random() * palette.length)];
        }

        update() {
            // Edge wrapping
            if (this.x < 0) this.x = W;
            if (this.x > W) this.x = 0;
            if (this.y < 0) this.y = H;
            if (this.y > H) this.y = 0;

            // Mouse repulsion
            if (MOUSE.x !== null) {
                const dx = this.x - MOUSE.x;
                const dy = this.y - MOUSE.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < MOUSE.radius) {
                    const force = (MOUSE.radius - dist) / MOUSE.radius * 0.015;
                    this.vx += dx * force;
                    this.vy += dy * force;
                }
            }

            // Damping
            this.vx *= 0.995;
            this.vy *= 0.995;

            this.x += this.vx;
            this.y += this.vy;
        }

        draw() {
            const { r: cr, g, b } = this.color;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${cr},${g},${b},0.7)`;
            ctx.fill();

            // Glow
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.r * 3, 0, Math.PI * 2);
            const grd = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.r * 3);
            grd.addColorStop(0, `rgba(${cr},${g},${b},0.15)`);
            grd.addColorStop(1, `rgba(${cr},${g},${b},0)`);
            ctx.fillStyle = grd;
            ctx.fill();
        }
    }

    // Init
    for (let i = 0; i < PARTICLE_COUNT; i++) {
        particles.push(new Particle());
    }

    function drawConnections() {
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < CONNECTION_DIST) {
                    const alpha = (1 - dist / CONNECTION_DIST) * 0.12;
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(0, 240, 255, ${alpha})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
    }

    function loop() {
        ctx.clearRect(0, 0, W, H);
        particles.forEach((p) => { p.update(); p.draw(); });
        drawConnections();
        requestAnimationFrame(loop);
    }

    loop();
})();
