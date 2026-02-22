const initVanta = () => {
  try {
    if (typeof VANTA !== 'undefined' && VANTA.NET) {
      VANTA.NET({
        el: '#vanta-bg',
        mouseControls: true,
        touchControls: true,
        gyroControls: false,
        minHeight: 200.00,
        minWidth: 200.00,
        scale: 1.00,
        scaleMobile: 1.00,
        color: 0x3a86ff,
        backgroundColor: 0x0a0a0a,
        points: 8.00,
        maxDistance: 25.00,
        spacing: 18.00
      });
    }
  } catch (err) {
    console.warn('Vanta initialization failed:', err);
  }
};

const animateOnScroll = () => {
  if (typeof gsap === 'undefined') return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        gsap.fromTo(entry.target,
          { opacity: 0, y: 40 },
          { opacity: 1, y: 0, duration: 0.8, ease: 'power2.out' }
        );
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.2 });

  document.querySelectorAll('.scroll-animate').forEach(el => observer.observe(el));
};

const handleCtaClick = (e) => {
  if (!e.target.closest('.cta')) return;
  try {
    gtag('event', 'click', {
      event_category: 'CTA',
      event_label: e.target.textContent.trim(),
      transport_type: 'beacon'
    });
  } catch (err) {
    console.warn('GTAG not available:', err);
  }
};

const init = () => {
  initVanta();
  animateOnScroll();
  document.addEventListener('click', handleCtaClick);
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}