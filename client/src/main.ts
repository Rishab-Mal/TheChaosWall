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

// ─── Comparison (WebSocket) ────────────────────────────────────────────────────

interface CompState {
  theta1: number;
  theta2: number;
  omega1: number;
  omega2: number;
}

interface CompFrame {
  t: number;
  actual: CompState;
  lstm: CompState;
  hnn: CompState;
}

interface CompCanvas {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  trail: Array<{ x: number; y: number }>;
  color: string;
}

const COMP_TRAIL_LEN = 80;

function initCompCanvas(id: string, color: string): CompCanvas | null {
  const canvas = document.getElementById(id) as HTMLCanvasElement | null;
  if (!canvas) return null;
  const ctx = canvas.getContext('2d')!;
  return { canvas, ctx, trail: [], color };
}

function resizeCompCanvas(c: CompCanvas) {
  c.canvas.width  = c.canvas.clientWidth;
  c.canvas.height = c.canvas.clientHeight;
}

function drawCompFrame(c: CompCanvas, state: CompState) {
  const w = c.canvas.width;
  const h = c.canvas.height;
  const scale = Math.min(w, h) * 0.27;
  const cx = w / 2;
  const cy = h * 0.32;

  c.ctx.fillStyle = '#0d0d1a';
  c.ctx.fillRect(0, 0, w, h);

  const x1 = cx + scale * Math.sin(state.theta1);
  const y1 = cy + scale * Math.cos(state.theta1);
  const x2 = x1 + scale * Math.sin(state.theta2);
  const y2 = y1 + scale * Math.cos(state.theta2);

  c.trail.push({ x: x2, y: y2 });
  if (c.trail.length > COMP_TRAIL_LEN) c.trail.shift();

  // Trail
  for (let i = 1; i < c.trail.length; i++) {
    const a = i / c.trail.length;
    c.ctx.beginPath();
    c.ctx.strokeStyle = hexAlpha(c.color, a * 0.9);
    c.ctx.lineWidth = 0.6 + a * 1.8;
    c.ctx.lineCap = 'round';
    c.ctx.moveTo(c.trail[i - 1].x, c.trail[i - 1].y);
    c.ctx.lineTo(c.trail[i].x, c.trail[i].y);
    c.ctx.stroke();
  }

  // Arms
  c.ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  c.ctx.lineWidth = 1.5;
  c.ctx.lineCap = 'round';
  c.ctx.beginPath(); c.ctx.moveTo(cx, cy); c.ctx.lineTo(x1, y1); c.ctx.stroke();
  c.ctx.beginPath(); c.ctx.moveTo(x1, y1); c.ctx.lineTo(x2, y2); c.ctx.stroke();

  // Mid bob
  c.ctx.beginPath();
  c.ctx.arc(x1, y1, 3, 0, Math.PI * 2);
  c.ctx.fillStyle = 'rgba(255,255,255,0.45)';
  c.ctx.fill();

  // Tip bob
  c.ctx.beginPath();
  c.ctx.arc(x2, y2, 5, 0, Math.PI * 2);
  c.ctx.fillStyle = c.color;
  c.ctx.shadowColor = c.color;
  c.ctx.shadowBlur = 10;
  c.ctx.fill();
  c.ctx.shadowBlur = 0;

  // Pivot
  c.ctx.beginPath();
  c.ctx.arc(cx, cy, 3, 0, Math.PI * 2);
  c.ctx.fillStyle = '#fff';
  c.ctx.fill();
}

function setWsStatus(online: boolean, text: string) {
  const dot  = document.getElementById('ws-dot');
  const label = document.getElementById('ws-status-text');
  if (dot)   dot.className = 'status-dot ' + (online ? 'online' : 'offline');
  if (label) label.textContent = text;
}

function initComparison() {
  const compActual = initCompCanvas('canvas-actual', '#4ade80');
  const compLstm   = initCompCanvas('canvas-lstm',   '#a78bfa');
  const compHnn    = initCompCanvas('canvas-hnn',    '#22d3ee');
  if (!compActual || !compLstm || !compHnn) return;

  const comps = [compActual, compLstm, compHnn];
  comps.forEach(resizeCompCanvas);
  window.addEventListener('resize', () => comps.forEach(resizeCompCanvas));

  function connect() {
    const ws = new WebSocket('ws://localhost:8000/ws');

    ws.onopen = () => setWsStatus(true, 'Connected — streaming frames');

    ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data) as CompFrame & { error?: string };
      if (data.error) { setWsStatus(false, data.error); return; }
      drawCompFrame(compActual, data.actual);
      drawCompFrame(compLstm,   data.lstm);
      drawCompFrame(compHnn,    data.hnn);
    };

    ws.onclose  = () => {
      setWsStatus(false, 'Disconnected — retrying in 3 s');
      setTimeout(connect, 3000);
    };

    ws.onerror  = () => setWsStatus(false, 'Backend offline — run: uvicorn project.data.stream_app:app');
  }

  connect();
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
  initComparison();
});
