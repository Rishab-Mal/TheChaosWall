import { PendulumState, rk4Step, getPositions } from './pendulum';

// ─── Config ───────────────────────────────────────────────────────────────────

const PALETTE = ['#a78bfa', '#22d3ee', '#fb923c', '#4ade80', '#f472b6'];
const NUM_PENDULUMS = 5;
const TRAIL_LEN = 450;
const DT = 0.016;
const SUBSTEPS = 3;
const FADE_ALPHA = 0.18; // background fade opacity each frame

// ─── Types ────────────────────────────────────────────────────────────────────

interface Sim {
  state: PendulumState;
  // Ring buffer: alternating x,y pairs
  trailBuf: Float32Array;
  trailHead: number;
  trailLen: number;
  color: string;
}

// ─── State ────────────────────────────────────────────────────────────────────

let canvas: HTMLCanvasElement;
let ctx: CanvasRenderingContext2D;
let sims: Sim[] = [];
let armScale = 1;
let ox = 0;
let oy = 0;

// ─── Init ─────────────────────────────────────────────────────────────────────

function makeSims(baseA1: number, baseA2: number): Sim[] {
  return Array.from({ length: NUM_PENDULUMS }, (_, i) => ({
    state: {
      theta1: baseA1 + i * 1e-4,
      theta2: baseA2 + i * 1e-4,
      omega1: 0,
      omega2: 0,
    },
    trailBuf: new Float32Array(TRAIL_LEN * 2),
    trailHead: 0,
    trailLen: 0,
    color: PALETTE[i],
  }));
}

function resize() {
  // Match CSS size
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;

  const w = canvas.width;
  const h = canvas.height;
  armScale = Math.min(w, h) * 0.23;
  ox = w / 2;
  oy = h * 0.30;

  // Repaint background immediately so no flash
  ctx.fillStyle = '#06060f';
  ctx.fillRect(0, 0, w, h);
}

// ─── Simulation step ──────────────────────────────────────────────────────────

function step() {
  for (const sim of sims) {
    for (let s = 0; s < SUBSTEPS; s++) {
      sim.state = rk4Step(sim.state, DT / SUBSTEPS);
    }
    const { x2, y2 } = getPositions(sim.state, ox, oy, armScale);
    const idx = sim.trailHead * 2;
    sim.trailBuf[idx]     = x2;
    sim.trailBuf[idx + 1] = y2;
    sim.trailHead = (sim.trailHead + 1) % TRAIL_LEN;
    if (sim.trailLen < TRAIL_LEN) sim.trailLen++;
  }
}

// ─── Render ───────────────────────────────────────────────────────────────────

function hexAlpha(hex: string, a: number): string {
  // a in [0,1], append 2-digit hex alpha to a 6-char hex color
  return hex + Math.round(a * 255).toString(16).padStart(2, '0');
}

function draw() {
  const w = canvas.width;
  const h = canvas.height;

  // Ghosting fade
  ctx.fillStyle = `rgba(6,6,15,${FADE_ALPHA})`;
  ctx.fillRect(0, 0, w, h);

  // ── Trails ──
  for (const sim of sims) {
    if (sim.trailLen < 2) continue;
    for (let i = 1; i < sim.trailLen; i++) {
      const t = i / sim.trailLen;
      const curr = ((sim.trailHead - sim.trailLen + i - 1 + TRAIL_LEN) % TRAIL_LEN) * 2;
      const next = ((sim.trailHead - sim.trailLen + i     + TRAIL_LEN) % TRAIL_LEN) * 2;

      ctx.beginPath();
      ctx.strokeStyle = hexAlpha(sim.color, t * 0.85);
      ctx.lineWidth = 0.8 + t * 1.6;
      ctx.lineCap = 'round';
      ctx.moveTo(sim.trailBuf[curr], sim.trailBuf[curr + 1]);
      ctx.lineTo(sim.trailBuf[next], sim.trailBuf[next + 1]);
      ctx.stroke();
    }
  }

  // ── Arms & bobs ──
  for (const sim of sims) {
    const { x1, y1, x2, y2 } = getPositions(sim.state, ox, oy, armScale);

    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.lineWidth = 1.5;
    ctx.lineCap = 'round';

    ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();

    // Mid bob
    ctx.beginPath();
    ctx.arc(x1, y1, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.55)';
    ctx.fill();

    // Tip bob (colored)
    ctx.beginPath();
    ctx.arc(x2, y2, 5.5, 0, Math.PI * 2);
    ctx.fillStyle = sim.color;
    ctx.shadowColor = sim.color;
    ctx.shadowBlur = 8;
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  // Pivot
  ctx.beginPath();
  ctx.arc(ox, oy, 4, 0, Math.PI * 2);
  ctx.fillStyle = '#fff';
  ctx.fill();
}

// ─── Loop ─────────────────────────────────────────────────────────────────────

function animate() {
  step();
  draw();
  requestAnimationFrame(animate);
}

// ─── Smooth scroll for nav links ──────────────────────────────────────────────

function initNav() {
  document.querySelectorAll<HTMLAnchorElement>('a[href^="#"]').forEach((a) => {
    a.addEventListener('click', (e) => {
      const href = a.getAttribute('href');
      if (!href) return;
      const target = document.querySelector(href);
      if (!target) return;
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth' });
    });
  });
}

// ─── Bootstrap ────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  canvas = document.getElementById('pendulum-canvas') as HTMLCanvasElement;
  ctx = canvas.getContext('2d')!;

  resize();
  window.addEventListener('resize', () => {
    resize();
    // Clear trails on resize to avoid jumps
    for (const sim of sims) {
      sim.trailLen = 0;
      sim.trailHead = 0;
    }
  });

  canvas.addEventListener('click', () => {
    const a1 = Math.PI * (0.3 + Math.random() * 1.2);
    const a2 = Math.PI * (0.3 + Math.random() * 1.2);
    sims = makeSims(a1, a2);
  });

  sims = makeSims(Math.PI * 0.75, Math.PI * 0.85);
  initNav();
  animate();
});
