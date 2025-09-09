// space-bg.js
(() => {
  const canvas = document.getElementById('space-canvas');
  const ctx = canvas.getContext('2d', { alpha: true });

  // Device pixel ratio scaling for crisp stars
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(window.innerWidth * dpr);
    canvas.height = Math.floor(window.innerHeight * dpr);
    canvas.style.width = window.innerWidth + 'px';
    canvas.style.height = window.innerHeight + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener('resize', resize);
  resize();

  // Config
  const NUM_STARS = Math.round((window.innerWidth * window.innerHeight) / 4000);
  const stars = [];

  for (let i = 0; i < NUM_STARS; i++) {
    stars.push({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      r: Math.random() * 1.2 + 0.4,
      speed: 0.2 + Math.random() * 0.4, // px/frame
      phase: Math.random() * Math.PI * 2
    });
  }

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // subtle dark gradient
    const g = ctx.createLinearGradient(0, 0, 0, canvas.height);
    g.addColorStop(0, 'rgba(10,15,25,1)');
    g.addColorStop(1, 'rgba(5,8,15,1)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let s of stars) {
      // Move star downward
      s.y += s.speed;
      if (s.y > window.innerHeight) {
        s.y = -2; // respawn at top
        s.x = Math.random() * window.innerWidth;
      }

      // Twinkle
      s.phase += 0.05;
      const tw = (Math.sin(s.phase) + 1) * 0.5; // 0..1
      const brightness = 0.6 + tw * 0.4;

      ctx.beginPath();
      ctx.fillStyle = `rgba(255,255,255,${brightness})`;
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fill();
    }

    requestAnimationFrame(animate);
  }

  animate();
})();
