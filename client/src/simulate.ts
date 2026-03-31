// ─── Types ────────────────────────────────────────────────────────────────────

interface State {
  theta1: number; theta2: number;
  omega1: number; omega2: number;
}
interface Frame { t: number; actual: State; lstm: State; hnn: State; error?: string; }
interface Panel {
  canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D;
  trail: Array<{ x: number; y: number }>; color: string;
  timeEl: HTMLElement | null;
}
interface Series { label: string; color: string; data: number[]; }

// ─── Constants ────────────────────────────────────────────────────────────────

const N_SIM_STEPS = 200;
const TRAIL_LEN   = 90;
const G = 9.81;

// ─── Physics ──────────────────────────────────────────────────────────────────

/** Total mechanical energy (m1=m2=l1=l2=1). */
function energy(s: State): number {
  const T = s.omega1 ** 2
          + 0.5 * s.omega2 ** 2
          + s.omega1 * s.omega2 * Math.cos(s.theta1 - s.theta2);
  const V = -2 * G * Math.cos(s.theta1) - G * Math.cos(s.theta2);
  return T + V;
}

// ─── Pendulum canvas ──────────────────────────────────────────────────────────

function hexAlpha(hex: string, a: number): string {
  return hex + Math.round(a * 255).toString(16).padStart(2, '0');
}

function makePanel(canvasId: string, timeId: string, color: string): Panel | null {
  const canvas = document.getElementById(canvasId) as HTMLCanvasElement | null;
  if (!canvas) return null;
  return { canvas, ctx: canvas.getContext('2d')!, trail: [], color, timeEl: document.getElementById(timeId) };
}

function resizePanel(p: Panel) {
  p.canvas.width  = p.canvas.clientWidth;
  p.canvas.height = p.canvas.clientHeight;
}

function clearPanel(p: Panel) {
  p.trail = [];
  const { ctx, canvas } = p;
  ctx.fillStyle = '#0d0d1a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (p.timeEl) p.timeEl.textContent = 't = —';
}

function drawPanel(p: Panel, state: State, t: number) {
  const { canvas, ctx, color } = p;
  const w = canvas.width, h = canvas.height;
  const scale = Math.min(w, h) * 0.27;
  const cx = w / 2, cy = h * 0.32;

  ctx.fillStyle = '#0d0d1a';
  ctx.fillRect(0, 0, w, h);

  const x1 = cx + scale * Math.sin(state.theta1);
  const y1 = cy + scale * Math.cos(state.theta1);
  const x2 = x1 + scale * Math.sin(state.theta2);
  const y2 = y1 + scale * Math.cos(state.theta2);

  p.trail.push({ x: x2, y: y2 });
  if (p.trail.length > TRAIL_LEN) p.trail.shift();

  for (let i = 1; i < p.trail.length; i++) {
    const a = i / p.trail.length;
    ctx.beginPath();
    ctx.strokeStyle = hexAlpha(color, a * 0.9);
    ctx.lineWidth = 0.6 + a * 2;
    ctx.lineCap = 'round';
    ctx.moveTo(p.trail[i - 1].x, p.trail[i - 1].y);
    ctx.lineTo(p.trail[i].x, p.trail[i].y);
    ctx.stroke();
  }

  ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  ctx.lineWidth = 1.5; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(cx, cy);   ctx.lineTo(x1, y1); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();

  ctx.beginPath(); ctx.arc(x1, y1, 3, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255,255,255,0.45)'; ctx.fill();

  ctx.beginPath(); ctx.arc(x2, y2, 5, 0, Math.PI * 2);
  ctx.fillStyle = color; ctx.shadowColor = color; ctx.shadowBlur = 12; ctx.fill();
  ctx.shadowBlur = 0;

  ctx.beginPath(); ctx.arc(cx, cy, 3, 0, Math.PI * 2);
  ctx.fillStyle = '#fff'; ctx.fill();

  if (p.timeEl) p.timeEl.textContent = `t = ${t.toFixed(2)} s`;
}

// ─── Chart helpers ────────────────────────────────────────────────────────────

const PAD = { top: 30, right: 18, bottom: 38, left: 50 };

function setupChart(canvas: HTMLCanvasElement): [CanvasRenderingContext2D, number, number] {
  const dpr = window.devicePixelRatio || 1;
  canvas.width  = canvas.clientWidth  * dpr;
  canvas.height = canvas.clientHeight * dpr;
  const ctx = canvas.getContext('2d')!;
  ctx.scale(dpr, dpr);
  return [ctx, canvas.clientWidth, canvas.clientHeight];
}

function autoBounds(series: Series[], fallbackSeries?: Series[]): [number, number] {
  const src = fallbackSeries ?? series;
  let lo = Infinity, hi = -Infinity;
  for (const s of src) for (const v of s.data) if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

function drawAxes(ctx: CanvasRenderingContext2D, w: number, h: number,
                  yMin: number, yMax: number, xMin: number, xMax: number,
                  title: string, yLabel: string, xLabel: string) {
  const { top, right, bottom, left } = PAD;
  const pw = w - left - right, ph = h - top - bottom;

  ctx.fillStyle = '#0d0d1a';
  ctx.fillRect(0, 0, w, h);

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.05)'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = top + (i / 4) * ph;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + pw, y); ctx.stroke();
  }
  for (let i = 0; i <= 5; i++) {
    const x = left + (i / 5) * pw;
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + ph); ctx.stroke();
  }

  // Zero line if in range
  if (yMin < 0 && yMax > 0) {
    const y0 = top + (1 - (0 - yMin) / (yMax - yMin)) * ph;
    ctx.strokeStyle = 'rgba(255,255,255,0.12)'; ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(left, y0); ctx.lineTo(left + pw, y0); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Axes
  ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, top); ctx.lineTo(left, top + ph); ctx.lineTo(left + pw, top + ph);
  ctx.stroke();

  // Y tick labels
  ctx.fillStyle = 'rgba(255,255,255,0.35)';
  ctx.font = '10px JetBrains Mono, monospace'; ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const v = yMin + (1 - i / 4) * (yMax - yMin);
    const y = top + (i / 4) * ph;
    ctx.fillText(v.toFixed(1), left - 5, y + 3.5);
  }

  // X tick labels
  ctx.textAlign = 'center';
  for (let i = 0; i <= 5; i++) {
    const v = xMin + (i / 5) * (xMax - xMin);
    const x = left + (i / 5) * pw;
    ctx.fillText(v.toFixed(1), x, top + ph + 14);
  }

  // Title
  ctx.fillStyle = 'rgba(255,255,255,0.65)';
  ctx.font = 'bold 11px Space Grotesk, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(title, left + pw / 2, 18);

  // Axis labels
  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  ctx.fillText(xLabel, left + pw / 2, h - 4);
  ctx.save();
  ctx.translate(11, top + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

function drawLineSeries(ctx: CanvasRenderingContext2D, w: number, h: number,
                        series: Series[], times: number[],
                        yMin: number, yMax: number, xMin: number, xMax: number) {
  const { top, right, bottom, left } = PAD;
  const pw = w - left - right, ph = h - top - bottom;
  const toX = (v: number) => left + ((v - xMin) / (xMax - xMin)) * pw;
  const toY = (v: number) => top  + (1 - (v - yMin) / (yMax - yMin)) * ph;

  ctx.save();
  ctx.beginPath(); ctx.rect(left, top, pw, ph); ctx.clip();

  for (const s of series) {
    ctx.beginPath(); ctx.strokeStyle = s.color; ctx.lineWidth = 1.5; ctx.lineJoin = 'round';
    let pen = false;
    for (let i = 0; i < times.length; i++) {
      const v = s.data[i];
      if (!isFinite(v)) { pen = false; continue; }
      const x = toX(times[i]), y = toY(v);
      if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function drawLegend(ctx: CanvasRenderingContext2D, series: Series[]) {
  const x0 = PAD.left + 8;
  let y = PAD.top + 14;
  ctx.font = '9px JetBrains Mono, monospace'; ctx.textAlign = 'left';
  for (const s of series) {
    ctx.fillStyle = s.color;
    ctx.fillRect(x0, y - 4, 14, 2);
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.fillText(s.label, x0 + 18, y);
    y += 13;
  }
}

// ─── Individual charts ────────────────────────────────────────────────────────

function renderTimeChart(canvasId: string, title: string, yLabel: string,
                         times: number[], series: Series[], refSeries?: Series[]) {
  const canvas = document.getElementById(canvasId) as HTMLCanvasElement | null;
  if (!canvas) return;
  const [ctx, w, h] = setupChart(canvas);
  const [yMin, yMax] = autoBounds(series, refSeries);
  const xMin = times[0], xMax = times[times.length - 1];
  drawAxes(ctx, w, h, yMin, yMax, xMin, xMax, title, yLabel, 'time (s)');
  drawLineSeries(ctx, w, h, series, times, yMin, yMax, xMin, xMax);
  drawLegend(ctx, series);
}

function renderPhaseChart(canvasId: string, frames: Frame[]) {
  const canvas = document.getElementById(canvasId) as HTMLCanvasElement | null;
  if (!canvas) return;
  const [ctx, w, h] = setupChart(canvas);
  const { top, right, bottom, left } = PAD;
  const pw = w - left - right, ph = h - top - bottom;

  // Bounds from actual only
  const axT = frames.map(f => f.actual.theta1);
  const axW = frames.map(f => f.actual.omega1);
  const [xMin, xMax] = autoBounds([{ label: '', color: '', data: axT }]);
  const [yMin, yMax] = autoBounds([{ label: '', color: '', data: axW }]);

  drawAxes(ctx, w, h, yMin, yMax, xMin, xMax, 'Phase portrait  θ₁ vs ω₁', 'ω₁ (rad/s)', 'θ₁ (rad)');

  const toX = (v: number) => left + ((v - xMin) / (xMax - xMin)) * pw;
  const toY = (v: number) => top  + (1 - (v - yMin) / (yMax - yMin)) * ph;

  ctx.save();
  ctx.beginPath(); ctx.rect(left, top, pw, ph); ctx.clip();

  const phaseSeries = [
    { label: 'actual', color: '#4ade80', theta: axT, omega: axW },
    { label: 'lstm',   color: '#a78bfa', theta: frames.map(f => f.lstm.theta1),   omega: frames.map(f => f.lstm.omega1) },
    { label: 'hnn',    color: '#22d3ee', theta: frames.map(f => f.hnn.theta1),    omega: frames.map(f => f.hnn.omega1) },
  ];

  for (const s of phaseSeries) {
    ctx.beginPath(); ctx.strokeStyle = s.color; ctx.lineWidth = 1.3; ctx.lineJoin = 'round';
    let pen = false;
    for (let i = 0; i < s.theta.length; i++) {
      const tx = s.theta[i], tw = s.omega[i];
      if (!isFinite(tx) || !isFinite(tw)) { pen = false; continue; }
      const x = toX(tx), y = toY(tw);
      if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.restore();

  drawLegend(ctx, phaseSeries.map(s => ({ label: s.label, color: s.color, data: [] })));
}

// ─── Main graph renderer ──────────────────────────────────────────────────────

function renderAllGraphs(frames: Frame[]) {
  const section = document.getElementById('graphs-section');
  if (section) { section.style.display = 'block'; section.scrollIntoView({ behavior: 'smooth', block: 'start' }); }

  const times  = frames.map(f => f.t);
  const actual = frames.map(f => f.actual);
  const lstm   = frames.map(f => f.lstm);
  const hnn    = frames.map(f => f.hnn);

  // θ₁(t)
  renderTimeChart('chart-theta1', 'θ₁ over time', 'θ₁ (rad)', times, [
    { label: 'actual', color: '#4ade80', data: actual.map(s => s.theta1) },
    { label: 'lstm',   color: '#a78bfa', data: lstm.map(s => s.theta1) },
    { label: 'hnn',    color: '#22d3ee', data: hnn.map(s => s.theta1) },
  ], [
    // Use only actual + hnn for y bounds so LSTM divergence doesn't crush the scale
    { label: '', color: '', data: actual.map(s => s.theta1) },
    { label: '', color: '', data: hnn.map(s => s.theta1) },
  ]);

  // θ₂(t)
  renderTimeChart('chart-theta2', 'θ₂ over time', 'θ₂ (rad)', times, [
    { label: 'actual', color: '#4ade80', data: actual.map(s => s.theta2) },
    { label: 'lstm',   color: '#a78bfa', data: lstm.map(s => s.theta2) },
    { label: 'hnn',    color: '#22d3ee', data: hnn.map(s => s.theta2) },
  ], [
    { label: '', color: '', data: actual.map(s => s.theta2) },
    { label: '', color: '', data: hnn.map(s => s.theta2) },
  ]);

  // Energy E(t)
  renderTimeChart('chart-energy', 'Total energy E(t)', 'E (J)', times, [
    { label: 'actual', color: '#4ade80', data: actual.map(energy) },
    { label: 'lstm',   color: '#a78bfa', data: lstm.map(energy) },
    { label: 'hnn',    color: '#22d3ee', data: hnn.map(energy) },
  ], [
    { label: '', color: '', data: actual.map(energy) },
    { label: '', color: '', data: hnn.map(energy) },
  ]);

  // Phase portrait
  renderPhaseChart('chart-phase', frames);
}

// ─── Status helpers ───────────────────────────────────────────────────────────

function setStatus(dot: HTMLElement | null, text: HTMLElement | null,
                   cls: 'online' | 'offline' | 'loading', msg: string) {
  if (dot)  dot.className = `status-dot ${cls}`;
  if (text) text.textContent = msg;
}

// ─── Bootstrap ────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  const panelActual = makePanel('canvas-actual', 'time-actual', '#4ade80');
  const panelLstm   = makePanel('canvas-lstm',   'time-lstm',   '#a78bfa');
  const panelHnn    = makePanel('canvas-hnn',     'time-hnn',    '#22d3ee');
  if (!panelActual || !panelLstm || !panelHnn) return;
  const panels = [panelActual, panelLstm, panelHnn];

  panels.forEach(resizePanel);
  window.addEventListener('resize', () => panels.forEach(resizePanel));

  const sliderT1 = document.getElementById('slider-t1') as HTMLInputElement;
  const sliderT2 = document.getElementById('slider-t2') as HTMLInputElement;
  const valT1    = document.getElementById('val-t1');
  const valT2    = document.getElementById('val-t2');
  const btnSim   = document.getElementById('btn-simulate') as HTMLButtonElement;
  const btnReset = document.getElementById('btn-reset')    as HTMLButtonElement;
  const wsDot    = document.getElementById('ws-dot');
  const wsText   = document.getElementById('ws-status-text');

  sliderT1.addEventListener('input', () => { if (valT1) valT1.textContent = `${sliderT1.value}°`; });
  sliderT2.addEventListener('input', () => { if (valT2) valT2.textContent = `${sliderT2.value}°`; });

  let ws: WebSocket | null = null;
  let collectedFrames: Frame[] = [];
  let graphsRendered = false;

  function reset() {
    if (ws) { ws.onclose = null; ws.onerror = null; ws.onmessage = null; ws.close(); ws = null; }
    collectedFrames = [];
    graphsRendered = false;
    panels.forEach(clearPanel);
    const gs = document.getElementById('graphs-section');
    if (gs) gs.style.display = 'none';
    btnSim.disabled   = false;
    btnReset.disabled = true;
    setStatus(wsDot, wsText, 'offline', 'Ready');
  }

  btnReset.addEventListener('click', reset);

  btnSim.addEventListener('click', () => {
    if (ws) { ws.onclose = null; ws.close(); }
    collectedFrames = [];
    graphsRendered = false;
    panels.forEach(clearPanel);
    const gs = document.getElementById('graphs-section');
    if (gs) gs.style.display = 'none';

    const theta1 = parseFloat(sliderT1.value) * (Math.PI / 180);
    const theta2 = parseFloat(sliderT2.value) * (Math.PI / 180);

    setStatus(wsDot, wsText, 'loading', 'Computing trajectories…');
    btnSim.disabled   = true;
    btnReset.disabled = false;

    ws = new WebSocket('ws://localhost:8000/ws/simulate');

    ws.onopen = () => {
      ws!.send(JSON.stringify({ theta1, theta2, omega1: 0, omega2: 0 }));
      setStatus(wsDot, wsText, 'online', 'Streaming…');
    };

    ws.onmessage = (ev) => {
      const frame = JSON.parse(ev.data) as Frame;
      if (frame.error) {
        setStatus(wsDot, wsText, 'offline', frame.error);
        btnSim.disabled = false;
        return;
      }

      drawPanel(panelActual, frame.actual, frame.t);
      drawPanel(panelLstm,   frame.lstm,   frame.t);
      drawPanel(panelHnn,    frame.hnn,    frame.t);

      // Collect first pass for graphs
      if (collectedFrames.length < N_SIM_STEPS) {
        collectedFrames.push(frame);
        if (collectedFrames.length === N_SIM_STEPS && !graphsRendered) {
          graphsRendered = true;
          renderAllGraphs(collectedFrames);
        }
      }
    };

    ws.onclose = () => {
      setStatus(wsDot, wsText, 'offline', 'Disconnected');
      btnSim.disabled   = false;
      btnReset.disabled = false;
    };

    ws.onerror = () => {
      setStatus(wsDot, wsText, 'offline', 'Backend offline — run: uvicorn project.data.stream_app:app');
      btnSim.disabled   = false;
      btnReset.disabled = true;
    };
  });
});
