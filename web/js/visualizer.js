/**
 * AudioVisualizer - HTML5 Canvas Web Audio API visualizer (Red & White Theme).
 */

class AudioVisualizer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.audioCtx = null;
    this.analyser = null;
    this.source = null;
    this.audioElement = null;
    this.animationId = null;
    this.isPlaying = false;

    this.resize();
    window.addEventListener('resize', () => this.resize());
    this.drawIdleState();
  }

  resize() {
    if (!this.canvas) return;
    this.canvas.width = this.canvas.clientWidth * window.devicePixelRatio || 600;
    this.canvas.height = this.canvas.clientHeight * window.devicePixelRatio || 100;
    this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  }

  connect(audioElement) {
    this.audioElement = audioElement;

    audioElement.addEventListener('play', () => {
      this.initAudioContext();
      this.isPlaying = true;
      this.draw();
    });

    audioElement.addEventListener('pause', () => {
      this.isPlaying = false;
      this.drawIdleState();
    });

    audioElement.addEventListener('ended', () => {
      this.isPlaying = false;
      this.drawIdleState();
    });
  }

  initAudioContext() {
    if (this.audioCtx) return;
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    this.audioCtx = new AudioContext();
    this.analyser = this.audioCtx.createAnalyser();
    this.analyser.fftSize = 128;
    this.analyser.smoothingTimeConstant = 0.8;

    try {
      this.source = this.audioCtx.createMediaElementSource(this.audioElement);
      this.source.connect(this.analyser);
      this.analyser.connect(this.audioCtx.destination);
    } catch (e) {
      console.warn("Visualizer audio node hook:", e);
    }
  }

  drawIdleState() {
    if (!this.canvas) return;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    this.ctx.clearRect(0, 0, w, h);

    // Draw center line
    this.ctx.beginPath();
    this.ctx.moveTo(0, h / 2);
    this.ctx.lineTo(w, h / 2);
    this.ctx.strokeStyle = "rgba(220, 38, 38, 0.35)";
    this.ctx.lineWidth = 2;
    this.ctx.stroke();

    // Resting dots
    const numDots = 24;
    for (let i = 0; i < numDots; i++) {
      const x = (w / (numDots + 1)) * (i + 1);
      this.ctx.beginPath();
      this.ctx.arc(x, h / 2, 2.5, 0, Math.PI * 2);
      this.ctx.fillStyle = "rgba(239, 68, 68, 0.6)";
      this.ctx.fill();
    }
  }

  draw() {
    if (!this.isPlaying) return;
    this.animationId = requestAnimationFrame(() => this.draw());

    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    this.ctx.clearRect(0, 0, w, h);

    if (!this.analyser) {
      this.drawSyntheticWave(w, h);
      return;
    }

    const bufferLength = this.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    this.analyser.getByteFrequencyData(dataArray);

    const barWidth = (w / bufferLength) * 2.2;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * (h * 0.85);

      // Red & White Gradient
      const grad = this.ctx.createLinearGradient(0, h, 0, 0);
      grad.addColorStop(0, '#991b1b');
      grad.addColorStop(0.5, '#dc2626');
      grad.addColorStop(1, '#ffffff');

      this.ctx.fillStyle = grad;
      this.ctx.shadowBlur = 8;
      this.ctx.shadowColor = 'rgba(220, 38, 38, 0.5)';

      const y = h / 2 - barHeight / 2;
      this.ctx.fillRect(x, y, barWidth - 2, barHeight);

      x += barWidth;
    }
  }

  drawSyntheticWave(w, h) {
    const time = performance.now() * 0.003;
    this.ctx.beginPath();
    this.ctx.moveTo(0, h / 2);

    for (let x = 0; x < w; x += 4) {
      const y = h / 2 + Math.sin(x * 0.04 + time) * 18 * Math.cos(x * 0.01 + time);
      this.ctx.lineTo(x, y);
    }

    this.ctx.strokeStyle = "#ef4444";
    this.ctx.lineWidth = 3;
    this.ctx.shadowBlur = 10;
    this.ctx.shadowColor = "#dc2626";
    this.ctx.stroke();
  }
}

window.AudioVisualizer = AudioVisualizer;
