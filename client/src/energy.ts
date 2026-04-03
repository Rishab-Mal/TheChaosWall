// ── Types ──────────────────────────────────────────────────────────────────────
interface ModelSeries {
  t:      number[];
  energy: (number | null)[];
}
interface AllSeries {
  actual?: ModelSeries;
  hnn?:    ModelSeries;
  lstm?:   ModelSeries;
}

// ── Config ─────────────────────────────────────────────────────────────────────
const COL_ACTUAL = '#16a34a';
const COL_HNN    = '#0891b2';
const COL_LSTM   = '#7c3aed';
const PAD        = { top: 54, right: 32, bottom: 50, left: 72 };

// ── State ──────────────────────────────────────────────────────────────────────
let data: AllSeries  = {};
let E0:   number     = 0;
let ws:   WebSocket | null = null;

// ── Canvas ─────────────────────────────────────────────────────────────────────
let canvas!:  HTMLCanvasElement;
let ctx!:     CanvasRenderingContext2D;
let lastW = 0, lastH = 0;

// ── DOM ────────────────────────────────────────────────────────────────────────
let statusDot!:  HTMLElement;
let statusText!: HTMLElement;
let btnRun!:     HTMLButtonElement;
let btnReset!:   HTMLButtonElement;
let cardActual!: HTMLElement;
let cardHnn!:    HTMLElement;
let cardLstm!:   HTMLElement;

// ── Helpers ────────────────────────────────────────────────────────────────────
function deg(id: string): number {
  return parseFloat((document.getElementById(id) as HTMLInputElement).value) * (Math.PI / 180);
}
function setStatus(cls: 'offline' | 'loading' | 'online', msg: string) {
  statusDot.className  = `sdot ${cls}`;
  statusText.textContent = msg;
}

// ── Chart ──────────────────────────────────────────────────────────────────────
function drawChart() {
  const dpr = window.devicePixelRatio || 1;
  const cw  = canvas.clientWidth;
  const ch  = canvas.clientHeight;
  if (cw !== lastW || ch !== lastH) {
    canvas.width  = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
    lastW = cw; lastH = ch;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const { top, right, bottom, left } = PAD;
  const pw = cw - left - right;
  const ph = ch - top - bottom;

  ctx.fillStyle = '#f5f5f7';
  ctx.fillRect(0, 0, cw, ch);

  // Empty state
  if (!data.actual) {
    ctx.fillStyle = 'rgba(0,0,0,0.25)';
    ctx.font = '13px Space Grotesk, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Set initial conditions and click Run', cw / 2, ch / 2);
    return;
  }

  // ── Bounds: scale y to GT + HNN, let LSTM go off-chart ──────────────────────
  const refVals: number[] = [];
  for (const series of [data.actual, data.hnn]) {
    if (!series) continue;
    for (const v of series.energy) if (v !== null && isFinite(v)) refVals.push(v);
  }
  if (refVals.length === 0) return;

  const rawMin = Math.min(...refVals);
  const rawMax = Math.max(...refVals);
  const margin = Math.max((rawMax - rawMin) * 0.25, 1.5);
  const yMin   = rawMin - margin;
  const yMax   = rawMax + margin;

  const tArr  = data.actual.t;
  const xMin  = tArr[0] ?? 0;
  const xMax  = tArr[tArr.length - 1] ?? 1;

  const toX = (t: number) => left + ((t - xMin) / (xMax - xMin)) * pw;
  const toY = (v: number) => top  + (1 - (v - yMin) / (yMax - yMin)) * ph;

  // ── Grid ────────────────────────────────────────────────────────────────────
  ctx.strokeStyle = 'rgba(0,0,0,0.07)'; ctx.lineWidth = 1;
  for (let i = 0; i <= 6; i++) {
    const y = top + (i / 6) * ph;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + pw, y); ctx.stroke();
  }
  for (let i = 0; i <= 8; i++) {
    const x = left + (i / 8) * pw;
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + ph); ctx.stroke();
  }

  // ── E₀ reference line ───────────────────────────────────────────────────────
  if (E0 !== 0) {
    const y0 = toY(E0);
    if (y0 >= top && y0 <= top + ph) {
      ctx.strokeStyle = 'rgba(0,0,0,0.18)'; ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);
      ctx.beginPath(); ctx.moveTo(left, y0); ctx.lineTo(left + pw, y0); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(0,0,0,0.38)';
      ctx.font = '9px JetBrains Mono, monospace'; ctx.textAlign = 'left';
      ctx.fillText(`E₀ = ${E0.toFixed(2)} J`, left + 5, y0 - 4);
    }
  }

  // ── Axes ────────────────────────────────────────────────────────────────────
  ctx.strokeStyle = 'rgba(0,0,0,0.28)'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, top); ctx.lineTo(left, top + ph); ctx.lineTo(left + pw, top + ph);
  ctx.stroke();

  // Y ticks
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.font = '10px JetBrains Mono, monospace'; ctx.textAlign = 'right';
  for (let i = 0; i <= 6; i++) {
    const v = yMin + (1 - i / 6) * (yMax - yMin);
    const y = top + (i / 6) * ph;
    ctx.fillText(v.toFixed(1), left - 8, y + 4);
  }
  // X ticks
  ctx.textAlign = 'center';
  for (let i = 0; i <= 8; i++) {
    const v = xMin + (i / 8) * (xMax - xMin);
    const x = left + (i / 8) * pw;
    ctx.fillText(v.toFixed(0) + ' s', x, top + ph + 16);
  }

  // Title
  ctx.fillStyle = 'rgba(0,0,0,0.8)';
  ctx.font = 'bold 12px Space Grotesk, sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('Total mechanical energy  E(t) = T + V', left + pw / 2, 24);

  // Y-axis label
  ctx.fillStyle = 'rgba(0,0,0,0.45)';
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.save(); ctx.translate(16, top + ph / 2); ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center'; ctx.fillText('E  (Joules)', 0, 0); ctx.restore();

  // X-axis label
  ctx.fillStyle = 'rgba(0,0,0,0.45)';
  ctx.font = '10px JetBrains Mono, monospace'; ctx.textAlign = 'center';
  ctx.fillText('time (s)', left + pw / 2, ch - 6);

  // ── Data lines (clipped to plot area) ───────────────────────────────────────
  ctx.save();
  ctx.beginPath(); ctx.rect(left, top, pw, ph); ctx.clip();

  const drawLine = (series: ModelSeries, color: string, lw: number) => {
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = lw;
    ctx.lineJoin = 'round';
    let pen = false;
    for (let i = 0; i < series.t.length; i++) {
      const v = series.energy[i];
      if (v === null || !isFinite(v)) { pen = false; continue; }
      const x = toX(series.t[i]);
      const y = toY(v);
      if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };

  // Draw LSTM first (background), then HNN, then GT on top
  if (data.lstm)   drawLine(data.lstm,   COL_LSTM,   1.8);
  if (data.hnn)    drawLine(data.hnn,    COL_HNN,    2.2);
  if (data.actual) drawLine(data.actual, COL_ACTUAL, 2.2);
  ctx.restore();

  // ── Legend (top-right inside plot) ──────────────────────────────────────────
  const entries = [
    { label: 'Ground Truth', color: COL_ACTUAL, ready: !!data.actual },
    { label: 'HNN',          color: COL_HNN,    ready: !!data.hnn    },
    { label: 'LSTM',         color: COL_LSTM,   ready: !!data.lstm   },
  ].filter(e => e.ready);

  let lx = left + pw - 120, ly = top + 14;
  ctx.font = '10px Space Grotesk, sans-serif'; ctx.textAlign = 'left';
  for (const e of entries) {
    ctx.fillStyle = e.color;
    ctx.fillRect(lx, ly - 3, 18, 3);
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillText(e.label, lx + 22, ly + 1);
    ly += 17;
  }
}

// ── Stat cards ─────────────────────────────────────────────────────────────────
function lastValid(series: ModelSeries | undefined): number | null {
  if (!series) return null;
  for (let i = series.energy.length - 1; i >= 0; i--) {
    const v = series.energy[i];
    if (v !== null && isFinite(v)) return v;
  }
  return null;
}

function fmtDrift(finalE: number | null): string {
  if (finalE === null || E0 === 0) return '—';
  const delta = finalE - E0;
  const pct   = (Math.abs(delta) / Math.abs(E0)) * 100;
  const sign  = delta >= 0 ? '+' : '';
  return `${sign}${delta.toFixed(2)} J &nbsp;(${pct.toFixed(1)}%)`;
}

function updateCards() {
  const actualFinal = lastValid(data.actual);
  const hnnFinal    = lastValid(data.hnn);
  const lstmFinal   = lastValid(data.lstm);

  const steps = data.actual?.t.length ?? 0;
  const dur   = steps > 0 ? ((steps - 1) * 0.05).toFixed(0) : '—';

  cardActual.innerHTML =
    `<div class="card-val">${E0 !== 0 ? E0.toFixed(3) + ' J' : '—'}</div>
     <div class="card-sub">Initial energy E₀</div>
     <div class="card-drift">${data.actual ? 'Drift: ' + fmtDrift(actualFinal) : 'computing…'}</div>
     <div class="card-dur">${dur} s simulated</div>`;

  cardHnn.innerHTML =
    `<div class="card-val">${hnnFinal !== null ? hnnFinal.toFixed(3) + ' J' : '—'}</div>
     <div class="card-sub">Final energy</div>
     <div class="card-drift">${data.hnn ? 'Drift: ' + fmtDrift(hnnFinal) : 'computing…'}</div>`;

  const lstmNote = lstmFinal === null && data.lstm
    ? '<div class="card-warn">diverged to NaN/Inf</div>'
    : `<div class="card-drift">${data.lstm ? 'Drift: ' + fmtDrift(lstmFinal) : 'computing…'}</div>`;
  cardLstm.innerHTML =
    `<div class="card-val">${lstmFinal !== null ? lstmFinal.toFixed(2) + ' J' : (data.lstm ? '∞' : '—')}</div>
     <div class="card-sub">Final energy</div>
     ${lstmNote}`;
}

// ── WebSocket ──────────────────────────────────────────────────────────────────
function run() {
  if (ws) { ws.onclose = ws.onerror = ws.onmessage = null; ws.close(); }
  data = {}; E0 = 0;
  drawChart(); updateCards();

  const theta1  = deg('slider-t1');
  const theta2  = deg('slider-t2');
  const n_steps = parseInt((document.getElementById('steps-select') as HTMLSelectElement).value);

  btnRun.disabled   = true;
  btnReset.disabled = false;
  setStatus('loading', 'Connecting…');

  ws = new WebSocket('ws://localhost:8000/ws/energy');

  ws.onopen = () => {
    setStatus('loading', 'Sending request…');
    ws!.send(JSON.stringify({ theta1, theta2, omega1: 0, omega2: 0, n_steps }));
  };

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data) as {
      type: string; model?: string;
      t?: number[]; energy?: (number | null)[];
      msg?: string;
    };
    if (msg.type === 'status') {
      setStatus('loading', msg.msg ?? '');
    } else if (msg.type === 'batch' && msg.model && msg.t && msg.energy) {
      const series: ModelSeries = { t: msg.t, energy: msg.energy };
      if      (msg.model === 'actual') { data.actual = series; if (msg.energy[0] != null) E0 = msg.energy[0]; }
      else if (msg.model === 'hnn')    data.hnn    = series;
      else if (msg.model === 'lstm')   data.lstm   = series;
      drawChart(); updateCards();
    } else if (msg.type === 'done') {
      setStatus('online', `Done — ${data.actual?.t.length ?? 0} steps  ·  ${((data.actual?.t.length ?? 1) - 1) * 0.05} s`);
      btnRun.disabled = false;
    } else if (msg.type === 'error') {
      setStatus('offline', msg.msg ?? 'Error');
      btnRun.disabled = false; btnReset.disabled = true;
    }
  };

  ws.onclose = () => { btnRun.disabled = false; };
  ws.onerror = () => {
    setStatus('offline', 'Backend offline — run: uvicorn project.data.stream_app:app');
    btnRun.disabled = false; btnReset.disabled = true;
  };
}

function reset() {
  if (ws) { ws.onclose = ws.onerror = ws.onmessage = null; ws.close(); ws = null; }
  data = {}; E0 = 0;
  drawChart(); updateCards();
  setStatus('offline', 'Ready');
  btnRun.disabled = false; btnReset.disabled = true;
}

// ── Bootstrap ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  canvas     = document.getElementById('energy-chart') as HTMLCanvasElement;
  ctx        = canvas.getContext('2d')!;
  statusDot  = document.getElementById('ws-dot')!;
  statusText = document.getElementById('ws-status-text')!;
  btnRun     = document.getElementById('btn-run')   as HTMLButtonElement;
  btnReset   = document.getElementById('btn-reset') as HTMLButtonElement;
  cardActual = document.getElementById('card-actual')!;
  cardHnn    = document.getElementById('card-hnn')!;
  cardLstm   = document.getElementById('card-lstm')!;

  const sliderT1 = document.getElementById('slider-t1') as HTMLInputElement;
  const sliderT2 = document.getElementById('slider-t2') as HTMLInputElement;
  sliderT1.addEventListener('input', () => {
    document.getElementById('val-t1')!.textContent = sliderT1.value + '°';
  });
  sliderT2.addEventListener('input', () => {
    document.getElementById('val-t2')!.textContent = sliderT2.value + '°';
  });

  btnRun.addEventListener('click', run);
  btnReset.addEventListener('click', reset);
  window.addEventListener('resize', drawChart);
  drawChart(); updateCards();
});
