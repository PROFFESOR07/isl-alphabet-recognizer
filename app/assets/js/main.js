/**
 * Janmotsava '26 - Certificate Retrieval Script
 * GDSC NITC
 */

document.addEventListener('DOMContentLoaded', () => {
  initStarsCanvas();
  initFormListeners();
});

/* =========================================================================
   ✨ 1. CELESTIAL TWINKLING STARS CANVAS
   ========================================================================= */
function initStarsCanvas() {
  const canvas = document.getElementById('stars-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  
  let width, height;
  let stars = [];
  const STAR_COUNT = 90;

  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
    createStars();
  }

  function createStars() {
    stars = [];
    for (let i = 0; i < STAR_COUNT; i++) {
      stars.push({
        x: Math.random() * width,
        y: Math.random() * height,
        radius: Math.random() * 1.6 + 0.4,
        alpha: Math.random() * 0.8 + 0.2,
        speed: Math.random() * 0.02 + 0.005,
        color: Math.random() > 0.3 ? '#f5cf68' : '#60a5fa'
      });
    }
  }

  function animate() {
    ctx.clearRect(0, 0, width, height);
    stars.forEach(star => {
      star.alpha += star.speed;
      if (star.alpha > 1 || star.alpha < 0.1) {
        star.speed = -star.speed;
      }
      ctx.beginPath();
      ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
      ctx.fillStyle = star.color;
      ctx.globalAlpha = Math.max(0, Math.min(1, star.alpha));
      ctx.shadowBlur = 6;
      ctx.shadowColor = star.color;
      ctx.fill();
    });
    requestAnimationFrame(animate);
  }

  window.addEventListener('resize', resize);
  resize();
  animate();
}

/* =========================================================================
   🎵 2. BANSURI FLUTE AUDIO CONTROLLER
   ========================================================================= */
let isPlaying = false;
let audioElem = null;
let audioContext = null;

function toggleFluteAudio() {
  const btn = document.getElementById('audioToggleBtn');
  if (!audioElem) {
    audioElem = document.getElementById('fluteAudio');
  }

  if (!isPlaying) {
    if (audioElem) {
      audioElem.play()
        .then(() => {
          isPlaying = true;
          btn.classList.add('playing');
        })
        .catch(err => {
          console.log('Audio autoplay prevented, playing web audio tone fallback', err);
          playFallbackFluteMelody();
          isPlaying = true;
          btn.classList.add('playing');
        });
    } else {
      playFallbackFluteMelody();
      isPlaying = true;
      btn.classList.add('playing');
    }
  } else {
    if (audioElem) {
      audioElem.pause();
    }
    stopFallbackAudio();
    isPlaying = false;
    btn.classList.remove('playing');
  }
}

// Gentle flute-like pentatonic synthesis fallback
let synthNodes = [];
function playFallbackFluteMelody() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!audioContext) audioContext = new AudioCtx();
    if (audioContext.state === 'suspended') audioContext.resume();

    const notes = [293.66, 329.63, 369.99, 440.00, 493.88, 587.33]; // Raga Bhupali notes
    let step = 0;

    const interval = setInterval(() => {
      if (!isPlaying) {
        clearInterval(interval);
        return;
      }
      const osc = audioContext.createOscillator();
      const gain = audioContext.createGain();
      
      osc.type = 'sine';
      osc.frequency.setValueAtTime(notes[step % notes.length], audioContext.currentTime);
      
      gain.gain.setValueAtTime(0, audioContext.currentTime);
      gain.gain.linearRampToValueAtTime(0.08, audioContext.currentTime + 0.3);
      gain.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + 1.8);

      osc.connect(gain);
      gain.connect(audioContext.destination);

      osc.start();
      osc.stop(audioContext.currentTime + 1.9);
      step++;
    }, 1400);

    synthNodes.push(interval);
  } catch (e) {
    console.error(e);
  }
}

function stopFallbackAudio() {
  synthNodes.forEach(id => clearInterval(id));
  synthNodes = [];
}

/* =========================================================================
   🔍 3. CERTIFICATE RETRIEVAL & VERIFICATION
   ========================================================================= */
function initFormListeners() {
  const input = document.getElementById('rollInput');
  const btn = document.getElementById('retrieveBtn');

  if (input) {
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        retrieveCertificate();
      }
    });
  }

  if (btn) {
    btn.addEventListener('click', retrieveCertificate);
  }
}

function retrieveCertificate() {
  const input = document.getElementById('rollInput');
  const statusContainer = document.getElementById('statusContainer');
  const btn = document.getElementById('retrieveBtn');
  
  if (!input || !statusContainer) return;
  const rollNumber = input.value.trim().toUpperCase();

  if (!rollNumber) {
    statusContainer.innerHTML = `
      <div class="info-box" style="border-color: rgba(255, 107, 107, 0.5); background: rgba(50, 14, 28, 0.75);">
        <div class="info-icon" style="border-color: #ff6b6b; color: #ff6b6b;">!</div>
        <div class="info-text" style="color: #ffc9c9;">Please enter a valid <span class="info-highlight" style="color: #ffffff;">Roll Number</span> to proceed.</div>
      </div>
    `;
    input.focus();
    return;
  }

  // Disable UI and show Matki Searching Animation
  btn.disabled = true;
  statusContainer.innerHTML = `
    <div class="matki-anim-card">
      <div class="matki-sprite-stage">
        <img src="assets/images/matki_search.svg" alt="Searching Matki" class="matki-sprite-img">
      </div>
      <div class="matki-caption">Searching Sacred Records for ${rollNumber}...</div>
    </div>
  `;

  // Simulate verification delay
  setTimeout(() => {
    btn.disabled = false;
    triggerConfetti();

    // Render Success Certificate Card
    statusContainer.innerHTML = `
      <div class="status-showcase-card">
        <div class="status-img-wrap">
          <img src="assets/images/certificate_success.svg" alt="Verified Certificate" class="status-illustration">
        </div>
        <div class="roll-verified-badge">
          <span>✓</span>
          <span>CERTIFICATE VERIFIED</span>
        </div>
        <div class="success-roll-info">
          Roll No: <strong>${rollNumber}</strong>
        </div>
        <a href="#download" class="btn-get-certificate" onclick="downloadCertificate('${rollNumber}')">
          <span>📥</span>
          <span>Download Certificate</span>
        </a>
      </div>
    `;
  }, 1200);
}

function downloadCertificate(rollNumber) {
  // Alert or direct link to Google Drive / API endpoint
  alert(`Certificate for ${rollNumber} is being prepared for download!`);
}

function triggerConfetti() {
  if (typeof confetti === 'function') {
    confetti({
      particleCount: 80,
      spread: 70,
      origin: { y: 0.6 },
      colors: ['#f5cf68', '#38ef7d', '#4285f4', '#ffffff', '#e5ad2e']
    });
  }
}
