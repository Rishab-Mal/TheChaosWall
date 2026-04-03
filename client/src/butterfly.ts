import { PendulumState, rk4Step, getPositions } from './pendulum';

// ── Config ─────────────────────────────────────────────────────────────────────
const DT             = 0.016;
const SUBSTEPS       = 4;
const TRAIL_LEN      = 280;
const MAX_HIST       = 1800;        // 30 s at 60 fps
const CHAOS_THRESH   = 0.5;        // arm lengths — "fully diverged"
const DIV_THRESH     = 0.02;       // arm lengths — "started diverging"

const COL_A   = '#7c3aed';         // purple
const COL_B   = '#ea580c';         // orange
const COL_A_R = '124,58,237';
const COL_B_R = '234,88,12';

// ── Simulation state ───────────────────────────────────────────────────────────
let stateA!: PendulumState;
let stateB!: PendulumState;
let trailA: { x: number; y: number }[] = [];
let trailB: { x: number; y: number }[] = [];
let sepHistory:  number[] = [];
let timeHistory: number[] = [];
let simTime   = 0;
let chaosTime: number | null = null;
let running   = false;
let rafId     = 0;
let epsilon   = 0.001;

// ── Canvas geometry ────────────────────────────────────────────────────────────
let mainCanvas!:  HTMLCanvasElement;
let mainCtx!:     CanvasRenderingContext2D;
let chartCanvas!: HTMLCanvasElement;
let chartCtx!:    CanvasRenderingContext2D;
let cx = 0, cy = 0, armScale = 0;
let lastChartW = 0, lastChartH = 0;

// ── DOM refs ───────────────────────────────────────────────────────────────────
let elTime!:      HTMLElement;
let elSep!:       HTMLElement;
let elStatusDot!: HTMLElement;
let elStatus!:    HTMLElement;
let elChaosAt!:   HTMLElement;
let elBarFill!:   HTMLElement;
let btnStart!:    HTMLButtonElement;
let btnReset!:    HTMLButtonElement;

// ── Helpers ────────────────────────────────────────────────────────────────────
function toRad(id: string): number {
  return parseFloat((document.getElementById(id) as HTMLInputElement).value) * (Math.PI / 180);
}

function initSim() {
  const t1 = toRad('slider-t1');
  const t2 = toRad('slider-t2');
  stateA = { theta1: t1,           theta2: t2, omega1: 0, omega2: 0 };
  stateB = { theta1: t1 + epsilon, theta2: t2, omega1: 0, omega2: 0 };
  trailA = []; trailB = [];
  sepHistory = []; timeHistory = [];
  simTime = 0; chaosTime = null;
}

function resizeMain() {
  mainCanvas.width  = mainCanvas.clientWidth;
  mainCanvas.height = mainCanvas.clientHeight;
  armScale = Math.min(mainCanvas.width, mainCanvas.height) * 0.27;
  cx = mainCanvas.width  / 2;
  cy = mainCanvas.height * 0.35;
}

function getSep(): number {
  const a = getPositions(stateA, cx, cy, armScale);
  const b = getPositions(stateB, cx, cy, armScale);
  return Math.hypot(a.x2 - b.x2, a.y2 - b.y2) / armScale;
}

// ── Physics ────────────────────────────────────────────────────────────────────
function tick() {
  for (let i = 0; i < SUBSTEPS; i++) {
    stateA = rk4Step(stateA, DT / SUBSTEPS);
    stateB = rk4Step(stateB, DT / SUBSTEPS);
  }
  simTime += DT;
}

// ── Render: pendulum canvas ────────────────────────────────────────────────────
function drawTrail(
  ctx: CanvasRenderingContext2D,
  trail: { x: number; y: number }[],
  rgb: string
) {
  for (let i = 1; i < trail.length; i++) {
    const a = i / trail.length;
    ctx.beginPath();
    ctx.strokeStyle = `rgba(${rgb},${(a * 0.82).toFixed(2)})`;
    ctx.lineWidth = 0.4 + a * 2.4;
    ctx.lineCap = 'round';
    ctx.moveTo(trail[i - 1].x, trail[i - 1].y);
    ctx.lineTo(trail[i].x, trail[i].y);
    ctx.stroke();
  }
}

function drawPend(
  ctx: CanvasRenderingContext2D,
  pos: ReturnType<typeof getPositions>,
  color: string,
  rgb: string
) {
  ctx.strokeStyle = `rgba(${rgb},0.28)`;
  ctx.lineWidth = 2; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(cx, cy);       ctx.lineTo(pos.x1, pos.y1); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pos.x1, pos.y1); ctx.lineTo(pos.x2, pos.y2); ctx.stroke();

  ctx.beginPath(); ctx.arc(pos.x1, pos.y1, 4, 0, Math.PI * 2);
  ctx.fillStyle = `rgba(${rgb},0.4)`; ctx.fill();

  ctx.beginPath(); ctx.arc(pos.x2, pos.y2, 7, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.shadowColor = color; ctx.shadowBlur = 14;
  ctx.fill(); ctx.shadowBlur = 0;
}

function drawMain(sep: number) {
  const w = mainCanvas.width, h = mainCanvas.height;
  mainCtx.fillStyle = '#f5f5f7';
  mainCtx.fillRect(0, 0, w, h);

  const posA = getPositions(stateA, cx, cy, armScale);
  const posB = getPositions(stateB, cx, cy, armScale);

  trailA.push({ x: posA.x2, y: posA.y2 });
  trailB.push({ x: posB.x2, y: posB.y2 });
  if (trailA.length > TRAIL_LEN) trailA.shift();
  if (trailB.length > TRAIL_LEN) trailB.shift();

  drawTrail(mainCtx, trailA, COL_A_R);
  drawTrail(mainCtx, trailB, COL_B_R);
  drawPend(mainCtx, posA, COL_A, COL_A_R);
  drawPend(mainCtx, posB, COL_B, COL_B_R);

  // Dashed connector between tips — fades as they diverge
  const alpha = Math.max(0, 0.55 - sep * 1.0);
  if (alpha > 0.01) {
    mainCtx.beginPath();
    mainCtx.strokeStyle = `rgba(60,60,60,${alpha.toFixed(2)})`;
    mainCtx.lineWidth = 1;
    mainCtx.setLineDash([3, 4]);
    mainCtx.moveTo(posA.x2, posA.y2);
    mainCtx.lineTo(posB.x2, posB.y2);
    mainCtx.stroke();
    mainCtx.setLineDash([]);
  }

  // Pivot
  mainCtx.beginPath(); mainCtx.arc(cx, cy, 5, 0, Math.PI * 2);
  mainCtx.fillStyle = '#1d1d1f'; mainCtx.fill();
}

// ── Render: log-divergence chart ───────────────────────────────────────────────
const PAD = { top: 38, right: 22, bottom: 44, left: 62 };

function drawChart() {
  if (sepHistory.length < 2) return;

  const dpr = window.devicePixelRatio || 1;
  const cw = chartCanvas.clientWidth;
  const ch = chartCanvas.clientHeight;

  if (cw !== lastChartW || ch !== lastChartH) {
    chartCanvas.width  = cw * dpr;
    chartCanvas.height = ch * dpr;
    lastChartW = cw; lastChartH = ch;
  }
  const ctx = chartCtx;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const { top, right, bottom, left } = PAD;
  const pw = cw - left - right;
  const ph = ch - top - bottom;

  ctx.fillStyle = '#f5f5f7';
  ctx.fillRect(0, 0, cw, ch);

  const logEps = Math.log10(epsilon);
  const logVals = sepHistory.map(d => d > 1e-9 ? Math.log10(d) : -9);
  const valid = logVals.filter(isFinite);

  const yMin = Math.min(logEps - 0.5, Math.min(...valid));
  const yMax = Math.max(Math.log10(2.5), Math.max(...valid));
  const xMin = timeHistory[0] ?? 0;
  const xMax = Math.max(timeHistory[timeHistory.length - 1] ?? 1, xMin + 1);

  const toX = (t: number) => left + ((t - xMin) / (xMax - xMin)) * pw;
  const toY = (v: number) => top  + (1 - (v - yMin) / (yMax - yMin)) * ph;

  // Grid
  ctx.strokeStyle = 'rgba(0,0,0,0.07)'; ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = top + (i / 5) * ph;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + pw, y); ctx.stroke();
  }
  for (let i = 0; i <= 6; i++) {
    const x = left + (i / 6) * pw;
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + ph); ctx.stroke();
  }

  // ε reference line
  const yEps = toY(logEps);
  if (yEps >= top && yEps <= top + ph) {
    ctx.strokeStyle = 'rgba(0,0,0,0.22)';
    ctx.lineWidth = 1; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(left, yEps); ctx.lineTo(left + pw, yEps); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(0,0,0,0.4)';
    ctx.font = '9px JetBrains Mono, monospace'; ctx.textAlign = 'left';
    ctx.fillText('initial ε', left + 6, yEps - 4);
  }

  // Chaos threshold line
  const yThresh = toY(Math.log10(CHAOS_THRESH));
  if (yThresh >= top && yThresh <= top + ph) {
    ctx.strokeStyle = 'rgba(220,50,50,0.4)';
    ctx.lineWidth = 1; ctx.setLineDash([3, 5]);
    ctx.beginPath(); ctx.moveTo(left, yThresh); ctx.lineTo(left + pw, yThresh); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(200,30,30,0.55)';
    ctx.font = '9px JetBrains Mono, monospace'; ctx.textAlign = 'right';
    ctx.fillText('chaos threshold', left + pw - 5, yThresh - 4);
  }

  // Axes
  ctx.strokeStyle = 'rgba(0,0,0,0.25)'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, top); ctx.lineTo(left, top + ph); ctx.lineTo(left + pw, top + ph);
  ctx.stroke();

  // Y tick labels
  ctx.fillStyle = 'rgba(0,0,0,0.5)';
  ctx.font = '10px JetBrains Mono, monospace'; ctx.textAlign = 'right';
  for (let i = 0; i <= 5; i++) {
    const v = yMin + (1 - i / 5) * (yMax - yMin);
    const y = top + (i / 5) * ph;
    ctx.fillText(`10^${v.toFixed(1)}`, left - 5, y + 4);
  }

  // X tick labels
  ctx.textAlign = 'center';
  for (let i = 0; i <= 6; i++) {
    const v = xMin + (i / 6) * (xMax - xMin);
    const x = left + (i / 6) * pw;
    ctx.fillText(v.toFixed(0) + ' s', x, top + ph + 16);
  }

  // Title
  ctx.fillStyle = 'rgba(0,0,0,0.8)';
  ctx.font = 'bold 11px Space Grotesk, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('Tip separation  (log₁₀ scale, arm lengths)', left + pw / 2, 20);

  // Y-axis label
  ctx.fillStyle = 'rgba(0,0,0,0.45)';
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.save();
  ctx.translate(13, top + ph / 2); ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center'; ctx.fillText('log₁₀(separation)', 0, 0);
  ctx.restore();

  // Divergence curve — colour shifts from purple (sync) to red (chaos) by value
  ctx.save();
  ctx.beginPath(); ctx.rect(left, top, pw, ph); ctx.clip();

  ctx.beginPath(); ctx.lineWidth = 2.2; ctx.lineJoin = 'round';
  let pen = false;
  for (let i = 0; i < logVals.length; i++) {
    if (!isFinite(logVals[i])) { pen = false; continue; }
    const x = toX(timeHistory[i]);
    const y = toY(logVals[i]);
    // colour by normalised divergence
    const norm = Math.max(0, Math.min(1, (sepHistory[i] - DIV_THRESH) / (CHAOS_THRESH - DIV_THRESH)));
    const r = Math.round(124 + norm * (220 - 124));
    const g = Math.round(58  + norm * (30  - 58));
    const b = Math.round(237 + norm * (30  - 237));
    ctx.strokeStyle = `rgb(${r},${g},${b})`;
    if (!pen) { ctx.beginPath(); ctx.moveTo(x, y); pen = true; } else { ctx.lineTo(x, y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x, y); }
  }
  ctx.stroke();
  ctx.restore();

  // Legend
  ctx.fillStyle = COL_A; ctx.fillRect(left + pw - 90, top + 8, 14, 3);
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.font = '9px JetBrains Mono, monospace'; ctx.textAlign = 'left';
  ctx.fillText('|A − B| tip', left + pw - 72, top + 12);
}

// ── DOM updates ────────────────────────────────────────────────────────────────
function updateDOM(sep: number) {
  elTime.textContent = `t = ${simTime.toFixed(2)} s`;
  elSep.textContent  = sep < 0.0001 ? '< 0.0001' : sep.toFixed(4);

  const pct = Math.min(100, (sep / 2.0) * 100);
  elBarFill.style.width = `${pct}%`;
  const hue = Math.max(0, 120 - pct * 1.2);
  elBarFill.style.background = `hsl(${hue}, 72%, 40%)`;

  let cls = 'sync', txt = 'In sync';
  if (sep >= CHAOS_THRESH) {
    cls = 'chaos'; txt = 'Fully chaotic';
    if (chaosTime === null) chaosTime = simTime;
  } else if (sep >= DIV_THRESH) {
    cls = 'diverging'; txt = 'Diverging';
  }
  elStatusDot.className = `sdot ${cls}`;
  elStatus.textContent  = txt;
  elStatus.className    = `status-text ${cls}`;
  elChaosAt.textContent = chaosTime !== null
    ? `chaos at t = ${chaosTime.toFixed(2)} s`
    : '';
}

// ── Initial canvas paint (before simulation starts) ────────────────────────────
function paintIdle() {
  const w = mainCanvas.width, h = mainCanvas.height;
  mainCtx.fillStyle = '#f5f5f7';
  mainCtx.fillRect(0, 0, w, h);
  const pos = getPositions(stateA, cx, cy, armScale);
  mainCtx.strokeStyle = `rgba(${COL_A_R},0.35)`;
  mainCtx.lineWidth = 2; mainCtx.lineCap = 'round';
  mainCtx.beginPath(); mainCtx.moveTo(cx, cy);       mainCtx.lineTo(pos.x1, pos.y1); mainCtx.stroke();
  mainCtx.beginPath(); mainCtx.moveTo(pos.x1, pos.y1); mainCtx.lineTo(pos.x2, pos.y2); mainCtx.stroke();
  mainCtx.beginPath(); mainCtx.arc(pos.x1, pos.y1, 4, 0, Math.PI * 2);
  mainCtx.fillStyle = `rgba(${COL_A_R},0.4)`; mainCtx.fill();
  mainCtx.beginPath(); mainCtx.arc(pos.x2, pos.y2, 7, 0, Math.PI * 2);
  mainCtx.fillStyle = COL_A; mainCtx.fill();
  mainCtx.beginPath(); mainCtx.arc(cx, cy, 5, 0, Math.PI * 2);
  mainCtx.fillStyle = '#1d1d1f'; mainCtx.fill();
}

// ── Animation loop ─────────────────────────────────────────────────────────────
function frame() {
  if (!running) return;
  tick();
  const sep = getSep();
  sepHistory.push(sep);
  timeHistory.push(simTime);
  if (sepHistory.length > MAX_HIST) { sepHistory.shift(); timeHistory.shift(); }
  drawMain(sep);
  drawChart();
  updateDOM(sep);
  rafId = requestAnimationFrame(frame);
}

// ── Controls ───────────────────────────────────────────────────────────────────
function start() {
  if (running) return;
  running = true;
  btnStart.disabled = true;
  btnReset.disabled = false;
  rafId = requestAnimationFrame(frame);
}

function reset() {
  running = false;
  cancelAnimationFrame(rafId);
  initSim();
  paintIdle();
  updateDOM(0);
  btnStart.disabled = false;
  btnReset.disabled = true;
}

// ── Bootstrap ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  mainCanvas  = document.getElementById('main-canvas')  as HTMLCanvasElement;
  chartCanvas = document.getElementById('chart-canvas') as HTMLCanvasElement;
  mainCtx  = mainCanvas.getContext('2d')!;
  chartCtx = chartCanvas.getContext('2d')!;

  elTime      = document.getElementById('el-time')!;
  elSep       = document.getElementById('el-sep')!;
  elStatusDot = document.getElementById('el-sdot')!;
  elStatus    = document.getElementById('el-status')!;
  elChaosAt   = document.getElementById('el-chaos')!;
  elBarFill   = document.getElementById('el-bar-fill')!;
  btnStart    = document.getElementById('btn-start') as HTMLButtonElement;
  btnReset    = document.getElementById('btn-reset') as HTMLButtonElement;

  // Slider display
  const sliderT1 = document.getElementById('slider-t1') as HTMLInputElement;
  const sliderT2 = document.getElementById('slider-t2') as HTMLInputElement;
  const valT1    = document.getElementById('val-t1')!;
  const valT2    = document.getElementById('val-t2')!;
  sliderT1.addEventListener('input', () => { valT1.textContent = `${sliderT1.value}°`; });
  sliderT2.addEventListener('input', () => { valT2.textContent = `${sliderT2.value}°`; });

  // Epsilon select
  const epsSelect = document.getElementById('eps-select') as HTMLSelectElement;
  const valEps    = document.getElementById('val-eps')!;
  epsSelect.addEventListener('change', () => {
    epsilon = parseFloat(epsSelect.value);
    valEps.textContent = epsSelect.options[epsSelect.selectedIndex].text.split(' ')[0];
  });

  btnStart.addEventListener('click', () => { initSim(); start(); });
  btnReset.addEventListener('click', reset);
  window.addEventListener('resize', () => { resizeMain(); paintIdle(); });

  resizeMain();
  initSim();
  paintIdle();
  updateDOM(0);
});
