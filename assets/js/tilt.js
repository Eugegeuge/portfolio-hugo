/**
 * 3D Tilt Effect for Cards
 * Adds a subtle parallax tilt and glare effect on hover.
 */

document.addEventListener('DOMContentLoaded', () => {
    const cards = document.querySelectorAll('.tilt-card');

    cards.forEach(card => {
        card.addEventListener('mousemove', handleMouseMove);
        card.addEventListener('mouseleave', handleMouseLeave);

        // Ensure parent has perspective if not set in CSS
        // card.style.transformStyle = 'preserve-3d';
    });

    function handleMouseMove(e) {
        const card = e.currentTarget;
        const rect = card.getBoundingClientRect();

        // Calculate mouse position relative to card center
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const centerX = rect.width / 2;
        const centerY = rect.height / 2;

        const rotateX = ((y - centerY) / centerY) * -5; // Max rotation deg
        const rotateY = ((x - centerX) / centerX) * 5;

        // Apply rotation
        card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`;

        // Glare Effect
        updateGlare(card, x, y);
    }

    function handleMouseLeave(e) {
        const card = e.currentTarget;
        card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) scale3d(1, 1, 1)';

        // Reset glare
        const glare = card.querySelector('.tilt-glare');
        if (glare) {
            glare.style.opacity = '0';
        }
    }

    function updateGlare(card, x, y) {
        let glare = card.querySelector('.tilt-glare');

        // Create glare if it doesn't exist
        if (!glare) {
            glare = document.createElement('div');
            glare.classList.add('tilt-glare');
            card.appendChild(glare);
        }

        // Calculate glare position
        glare.style.background = `radial-gradient(circle at ${x}px ${y}px, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0) 80%)`;
        glare.style.opacity = '1';
    }
});
