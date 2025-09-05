// Interactive Background Effects for Horizon AI

class InteractiveBackground {
    constructor() {
        this.mouseX = 0;
        this.mouseY = 0;
        this.cursorTrails = [];
        this.glowOrbs = [];
        this.particles = [];
        this.init();
    }

    init() {
        this.createFloatingParticles();
        this.createGridOverlay();
        this.createGlowOrbs();
        this.setupMouseInteraction();
        this.setupResizeHandler();
    }

    createFloatingParticles() {
        const particlesContainer = document.createElement('div');
        particlesContainer.className = 'floating-particles';
        document.body.appendChild(particlesContainer);

        // Create 15 particles
        for (let i = 0; i < 15; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            
            // Random properties
            const size = Math.random() * 6 + 2; // 2-8px
            const opacity = Math.random() * 0.6 + 0.2; // 0.2-0.8
            const color = this.getRandomColor();
            
            particle.style.width = `${size}px`;
            particle.style.height = `${size}px`;
            particle.style.background = color;
            particle.style.left = `${Math.random() * 100}%`;
            particle.style.animationDelay = `${Math.random() * 20}s`;
            particle.style.animationDuration = `${15 + Math.random() * 10}s`;
            
            particlesContainer.appendChild(particle);
        }
    }

    createGridOverlay() {
        const gridOverlay = document.createElement('div');
        gridOverlay.className = 'grid-overlay';
        document.body.appendChild(gridOverlay);
    }

    createGlowOrbs() {
        // Create 3 glow orbs
        for (let i = 0; i < 3; i++) {
            const orb = document.createElement('div');
            orb.className = 'glow-orb';
            document.body.appendChild(orb);
            this.glowOrbs.push(orb);
        }
    }

    setupMouseInteraction() {
        let lastTime = 0;
        const throttleDelay = 16; // ~60fps

        document.addEventListener('mousemove', (e) => {
            const currentTime = Date.now();
            if (currentTime - lastTime < throttleDelay) return;
            lastTime = currentTime;

            this.mouseX = e.clientX;
            this.mouseY = e.clientY;
            
            // Removed updateGlowOrbs() and createCursorTrail()
            this.updateBackgroundGradient();
        });

        document.addEventListener('mouseenter', () => {
            this.activateInteractiveElements();
        });

        document.addEventListener('mouseleave', () => {
            this.deactivateInteractiveElements();
        });

        // Removed click effects
    }

    updateGlowOrbs() {
        this.glowOrbs.forEach((orb, index) => {
            const delay = index * 100;
            const offsetX = (index - 1) * 50;
            const offsetY = (index - 1) * 30;
            
            setTimeout(() => {
                orb.style.left = `${this.mouseX + offsetX - 100}px`;
                orb.style.top = `${this.mouseY + offsetY - 100}px`;
                orb.classList.add('active');
            }, delay);
        });
    }

    createCursorTrail() {
        const trail = document.createElement('div');
        trail.className = 'cursor-trail active';
        trail.style.left = `${this.mouseX}px`;
        trail.style.top = `${this.mouseY}px`;
        
        document.body.appendChild(trail);
        
        // Remove trail after animation
        setTimeout(() => {
            trail.classList.remove('active');
            setTimeout(() => {
                if (trail.parentNode) {
                    trail.parentNode.removeChild(trail);
                }
            }, 1000);
        }, 200);
    }

    createClickRipple(x, y) {
        const ripple = document.createElement('div');
        ripple.style.position = 'fixed';
        ripple.style.left = `${x}px`;
        ripple.style.top = `${y}px`;
        ripple.style.width = '0px';
        ripple.style.height = '0px';
        ripple.style.border = '2px solid rgba(78, 205, 196, 0.6)';
        ripple.style.borderRadius = '50%';
        ripple.style.transform = 'translate(-50%, -50%)';
        ripple.style.pointerEvents = 'none';
        ripple.style.zIndex = '9998';
        ripple.style.animation = 'rippleEffect 0.6s ease-out forwards';
        
        document.body.appendChild(ripple);
        
        // Add ripple animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes rippleEffect {
                0% {
                    width: 0px;
                    height: 0px;
                    opacity: 1;
                }
                100% {
                    width: 100px;
                    height: 100px;
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
        
        setTimeout(() => {
            if (ripple.parentNode) {
                ripple.parentNode.removeChild(ripple);
            }
        }, 600);
    }

    updateBackgroundGradient() {
        const xPercent = (this.mouseX / window.innerWidth) * 100;
        const yPercent = (this.mouseY / window.innerHeight) * 100;
        
        // Update CSS custom properties for dynamic background
        document.documentElement.style.setProperty('--mouse-x', `${xPercent}%`);
        document.documentElement.style.setProperty('--mouse-y', `${yPercent}%`);
    }

    activateInteractiveElements() {
        // Enhance particles when mouse enters
        const particles = document.querySelectorAll('.particle');
        particles.forEach(particle => {
            particle.style.animationPlayState = 'running';
        });

        // Activate grid overlay
        const gridOverlay = document.querySelector('.grid-overlay');
        if (gridOverlay) {
            gridOverlay.style.opacity = '0.08';
        }
    }

    deactivateInteractiveElements() {
        // Hide glow orbs when mouse leaves
        this.glowOrbs.forEach(orb => {
            orb.classList.remove('active');
        });

        // Reset grid overlay
        const gridOverlay = document.querySelector('.grid-overlay');
        if (gridOverlay) {
            gridOverlay.style.opacity = '0.03';
        }
    }

    getRandomColor() {
        const colors = [
            'rgba(78, 205, 196, 0.6)',
            'rgba(69, 183, 209, 0.6)',
            'rgba(150, 206, 180, 0.6)',
            'rgba(255, 193, 7, 0.4)',
            'rgba(102, 126, 234, 0.5)'
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }

    setupResizeHandler() {
        window.addEventListener('resize', () => {
            // Reposition elements on resize
            this.updateGlowOrbs();
        });
    }

    // Add shooting stars effect
    createShootingStar() {
        const star = document.createElement('div');
        star.style.position = 'fixed';
        star.style.width = '2px';
        star.style.height = '2px';
        star.style.background = 'rgba(78, 205, 196, 0.8)';
        star.style.borderRadius = '50%';
        star.style.boxShadow = '0 0 10px rgba(78, 205, 196, 0.8)';
        star.style.pointerEvents = 'none';
        star.style.zIndex = '-1';
        
        // Random starting position
        const startX = Math.random() * window.innerWidth;
        const startY = -50;
        const endX = startX + (Math.random() * 200 - 100);
        const endY = window.innerHeight + 50;
        
        star.style.left = `${startX}px`;
        star.style.top = `${startY}px`;
        star.style.animation = `shootingStar 2s linear forwards`;
        
        // Add shooting star animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes shootingStar {
                0% {
                    transform: translateY(0) translateX(0);
                    opacity: 0;
                }
                10% {
                    opacity: 1;
                }
                90% {
                    opacity: 1;
                }
                100% {
                    transform: translateY(${endY - startY}px) translateX(${endX - startX}px);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(star);
        
        setTimeout(() => {
            if (star.parentNode) {
                star.parentNode.removeChild(star);
            }
        }, 2000);
    }

    // Start shooting stars at random intervals
    startShootingStars() {
        setInterval(() => {
            if (Math.random() < 0.3) { // 30% chance every interval
                this.createShootingStar();
            }
        }, 5000); // Check every 5 seconds
    }
}

// Initialize interactive background when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const interactiveBackground = new InteractiveBackground();
    
    // Start shooting stars after a delay
    setTimeout(() => {
        interactiveBackground.startShootingStars();
    }, 3000);
});
