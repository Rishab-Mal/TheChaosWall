import sys
import asyncio
import json
import numpy as np
import torch
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import polars as pl

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))

from Hamiltonian.model import HNN
from Hamiltonian.config import INPUT_DIM, HIDDEN_DIM, EPOCHS, RNN_HIDDEN, SEQ_LEN
from project.models.RNNModel import RNNModel

ACTUAL_PARQUET = root / "comparisons" / "actual.parquet"
RNN_PARQUET    = root / "comparisons" / "rnn_predicted.parquet"
HNN_PARQUET    = root / "comparisons" / "hnn_predicted.parquet"
RNN_PATH       = root / "models" / "rnn.pth"
HNN_PATH       = root / "models" / f"hnn_epoch_{EPOCHS}.pth"
COLS           = ["t", "theta1", "theta2", "theta1_dot", "theta2_dot"]
DT             = 0.05
N_STEPS        = 200
G              = 9.81
MAX_ENERGY_STEPS = 2400

# ── Load models once at startup ────────────────────────────────────────────────

_rnn: RNNModel | None       = None
_hnn: HNN | None            = None
_hnn_mean: torch.Tensor | None = None
_hnn_std:  torch.Tensor | None = None


def _load_models():
    global _rnn, _hnn, _hnn_mean, _hnn_std
    try:
        if RNN_PATH.exists():
            _rnn = RNNModel(input_size=INPUT_DIM, hidden_size=RNN_HIDDEN, output_size=INPUT_DIM)
            _rnn.load_state_dict(torch.load(str(RNN_PATH), map_location="cpu", weights_only=True))
            _rnn.eval()
            print(f"RNN loaded from {RNN_PATH.name}")
        if HNN_PATH.exists():
            ckpt = torch.load(str(HNN_PATH), map_location="cpu", weights_only=True)
            _hnn = HNN(input_size=INPUT_DIM, hidden_size=HIDDEN_DIM)
            _hnn.load_state_dict(ckpt["state_dict"])
            _hnn.eval()
            _hnn_mean = ckpt["state_mean"]
            _hnn_std  = ckpt["state_std"]
            print(f"HNN loaded from {HNN_PATH.name}")
    except Exception as e:
        print(f"Warning: model loading failed — {e}")


_load_models()


# ── Physics helpers ────────────────────────────────────────────────────────────

def _pendulum_derivs(state: np.ndarray) -> np.ndarray:
    """Standard double-pendulum equations (m1=m2=l1=l2=1)."""
    t1, t2, w1, w2 = state
    delta = t1 - t2
    denom = 3 - np.cos(2 * delta)
    dw1 = (
        -9.81 * (2 + 1) * np.sin(t1)
        - np.sin(t1 - 2 * t2) * 9.81
        - 2 * np.sin(delta) * (w2**2 + w1**2 * np.cos(delta))
    ) / denom
    dw2 = (
        2 * np.sin(delta) * (
            w1**2 * 2
            + 9.81 * 2 * np.cos(t1)
            + w2**2 * np.cos(delta)
        )
    ) / denom
    return np.array([w1, w2, dw1, dw2])


def _rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    k1 = _pendulum_derivs(state)
    k2 = _pendulum_derivs(state + 0.5 * dt * k1)
    k3 = _pendulum_derivs(state + 0.5 * dt * k2)
    k4 = _pendulum_derivs(state + dt * k3)
    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def _hnn_deriv(state: torch.Tensor) -> torch.Tensor:
    x_norm = (state.unsqueeze(0) - _hnn_mean) / _hnn_std
    d_norm = _hnn.time_derivatives(x_norm)
    return (d_norm * _hnn_std).squeeze(0).detach()


def _calc_energy(state) -> float | None:
    """Total mechanical energy  E = T + V  (m1=m2=l1=l2=1)."""
    try:
        t1, t2, w1, w2 = float(state[0]), float(state[1]), float(state[2]), float(state[3])
        if not all(map(np.isfinite, [t1, t2, w1, w2])):
            return None
        T = w1**2 + 0.5 * w2**2 + w1 * w2 * np.cos(t1 - t2)
        V = -2 * G * np.cos(t1) - G * np.cos(t2)
        e = T + V
        return float(e) if np.isfinite(e) else None
    except Exception:
        return None


def _gt_energy_series(theta1: float, theta2: float,
                      omega1: float, omega2: float, n: int) -> dict:
    state = np.array([theta1, theta2, omega1, omega2], dtype=np.float64)
    t_vals, e_vals = [], []
    for i in range(n):
        t_vals.append(round(i * DT, 3))
        e_vals.append(_calc_energy(state))
        state = _rk4_step(state, DT)
    return {"t": t_vals, "energy": e_vals}


def _hnn_energy_series(theta1: float, theta2: float,
                       omega1: float, omega2: float, n: int) -> dict:
    if _hnn is None:
        return {"t": [], "energy": []}
    s = torch.tensor([theta1, theta2, omega1, omega2], dtype=torch.float32)
    t_vals, e_vals = [], []
    for i in range(n):
        t_vals.append(round(i * DT, 3))
        e_vals.append(_calc_energy(s.numpy()))
        k1 = _hnn_deriv(s)
        k2 = _hnn_deriv(s + 0.5 * DT * k1)
        k3 = _hnn_deriv(s + 0.5 * DT * k2)
        k4 = _hnn_deriv(s + DT * k3)
        s  = (s + (DT / 6) * (k1 + 2 * k2 + 2 * k3 + k4)).detach()
    return {"t": t_vals, "energy": e_vals}


def _lstm_energy_series(theta1: float, theta2: float,
                        omega1: float, omega2: float, n: int) -> dict:
    if _rnn is None:
        return {"t": [], "energy": []}

    # Build ground-truth warmup window
    warmup = np.empty((SEQ_LEN, 4), dtype=np.float64)
    s = np.array([theta1, theta2, omega1, omega2], dtype=np.float64)
    for i in range(SEQ_LEN):
        warmup[i] = s
        s = _rk4_step(s, DT)

    t_vals, e_vals = [], []
    window = torch.tensor(warmup, dtype=torch.float32).unsqueeze(0)  # [1, SEQ_LEN, 4]
    diverged = False

    with torch.inference_mode():
        for i in range(n):
            t_vals.append(round(i * DT, 3))
            if i < SEQ_LEN:
                e_vals.append(_calc_energy(warmup[i]))
                continue
            if diverged:
                e_vals.append(None)
                continue
            out = _rnn(window).squeeze(0).numpy()
            # Angle unwrapping
            prev = window[0, -1, :2].numpy()
            for j in range(2):
                d = out[j] - prev[j]
                out[j] = prev[j] + (d + np.pi) % (2 * np.pi) - np.pi
            if not np.all(np.isfinite(out)):
                diverged = True
                e_vals.append(None)
            else:
                e_vals.append(_calc_energy(out))
                window = torch.cat(
                    [window[:, 1:, :],
                     torch.tensor(out[np.newaxis, np.newaxis], dtype=torch.float32)],
                    dim=1,
                )
    return {"t": t_vals, "energy": e_vals}


# ── Trajectory computation ─────────────────────────────────────────────────────

def _compute_trajectories(theta1: float, theta2: float,
                          omega1: float, omega2: float) -> list[dict]:
    init = np.array([theta1, theta2, omega1, omega2], dtype=np.float64)

    # Ground truth — RK4 physics
    gt = np.empty((N_STEPS, 4))
    gt[0] = init
    for i in range(1, N_STEPS):
        gt[i] = _rk4_step(gt[i - 1], DT)

    # LSTM — warm up with SEQ_LEN ground-truth steps, then autoregressive
    lstm = np.empty((N_STEPS, 4))
    lstm[:SEQ_LEN] = gt[:SEQ_LEN]
    window = torch.tensor(gt[:SEQ_LEN], dtype=torch.float32).unsqueeze(0)  # [1, 20, 4]
    with torch.inference_mode():
        for i in range(SEQ_LEN, N_STEPS):
            out = _rnn(window).squeeze(0).numpy()
            # Angle unwrapping to avoid 2π discontinuities
            prev = window[0, -1, :2].numpy()
            for j in range(2):
                d = out[j] - prev[j]
                out[j] = prev[j] + (d + np.pi) % (2 * np.pi) - np.pi
            lstm[i] = out
            window = torch.cat(
                [window[:, 1:, :],
                 torch.tensor(out[np.newaxis, np.newaxis], dtype=torch.float32)],
                dim=1,
            )

    # HNN — RK4 using learned Hamiltonian
    hnn = np.empty((N_STEPS, 4))
    s = torch.tensor(init, dtype=torch.float32)
    for i in range(N_STEPS):
        hnn[i] = s.numpy()
        k1 = _hnn_deriv(s)
        k2 = _hnn_deriv(s + 0.5 * DT * k1)
        k3 = _hnn_deriv(s + 0.5 * DT * k2)
        k4 = _hnn_deriv(s + DT * k3)
        s = (s + (DT / 6) * (k1 + 2 * k2 + 2 * k3 + k4)).detach()

    frames = []
    for i in range(N_STEPS):
        g, l, h = gt[i], lstm[i], hnn[i]
        frames.append({
            "t": round(i * DT, 4),
            "actual": {"theta1": float(g[0]), "theta2": float(g[1]),
                       "omega1": float(g[2]), "omega2": float(g[3])},
            "lstm":   {"theta1": float(l[0]), "theta2": float(l[1]),
                       "omega1": float(l[2]), "omega2": float(l[3])},
            "hnn":    {"theta1": float(h[0]), "theta2": float(h[1]),
                       "omega1": float(h[2]), "omega2": float(h[3])},
        })
    return frames


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return HTMLResponse("<html><body>WebSocket at <code>/ws</code> or <code>/ws/simulate</code></body></html>")


@app.websocket("/ws")
async def ws_precomputed(ws: WebSocket):
    """Stream the pre-computed comparison parquets (actual vs RNN vs HNN)."""
    await ws.accept()

    missing = [p.name for p in [ACTUAL_PARQUET, RNN_PARQUET, HNN_PARQUET] if not p.exists()]
    if missing:
        await ws.send_text(json.dumps({
            "error": f"Missing: {', '.join(missing)} — run compare scripts first"
        }))
        await ws.close()
        return

    actual = pl.read_parquet(str(ACTUAL_PARQUET)).sort("t").select(COLS)
    rnn    = pl.read_parquet(str(RNN_PARQUET)).sort("t").select(COLS)
    hnn    = pl.read_parquet(str(HNN_PARQUET)).sort("t").select(COLS)
    n = min(len(actual), len(rnn), len(hnn))
    frames = []
    for i in range(n):
        ar, rr, hr = actual.row(i, named=True), rnn.row(i, named=True), hnn.row(i, named=True)
        frames.append({
            "t": round(ar["t"], 4),
            "actual": {"theta1": ar["theta1"], "theta2": ar["theta2"],
                       "omega1": ar["theta1_dot"], "omega2": ar["theta2_dot"]},
            "lstm":   {"theta1": rr["theta1"], "theta2": rr["theta2"],
                       "omega1": rr["theta1_dot"], "omega2": rr["theta2_dot"]},
            "hnn":    {"theta1": hr["theta1"], "theta2": hr["theta2"],
                       "omega1": hr["theta1_dot"], "omega2": hr["theta2_dot"]},
        })

    try:
        while True:
            for frame in frames:
                await ws.send_text(json.dumps(frame))
                await asyncio.sleep(DT)
    except (WebSocketDisconnect, Exception):
        pass


@app.websocket("/ws/simulate")
async def ws_simulate(ws: WebSocket):
    """Accept initial conditions, compute trajectories, stream frames."""
    await ws.accept()

    if _rnn is None or _hnn is None:
        await ws.send_text(json.dumps({
            "error": "Models not loaded — check models/rnn.pth and models/hnn_epoch_200.pth exist"
        }))
        await ws.close()
        return

    try:
        raw = await ws.receive_text()
        p = json.loads(raw)
        theta1 = float(p["theta1"])
        theta2 = float(p["theta2"])
        omega1 = float(p.get("omega1", 0.0))
        omega2 = float(p.get("omega2", 0.0))
    except Exception:
        await ws.send_text(json.dumps({"error": "Invalid initial conditions"}))
        await ws.close()
        return

    # Compute in a thread so the event loop stays responsive
    frames = await asyncio.get_event_loop().run_in_executor(
        None, _compute_trajectories, theta1, theta2, omega1, omega2
    )

    try:
        while True:
            for frame in frames:
                await ws.send_text(json.dumps(frame))
                await asyncio.sleep(DT)
    except (WebSocketDisconnect, Exception):
        pass

@app.websocket("/ws/energy")
async def ws_energy(ws: WebSocket):
    """Long-horizon energy drift: streams one batch per model then closes."""
    await ws.accept()

    if _rnn is None or _hnn is None:
        await ws.send_text(json.dumps({
            "type": "error",
            "msg": "Models not loaded — check models/ directory"
        }))
        await ws.close()
        return

    try:
        raw    = await ws.receive_text()
        p      = json.loads(raw)
        theta1 = float(p["theta1"])
        theta2 = float(p["theta2"])
        omega1 = float(p.get("omega1", 0.0))
        omega2 = float(p.get("omega2", 0.0))
        n      = min(int(p.get("n_steps", 1200)), MAX_ENERGY_STEPS)
    except Exception:
        await ws.send_text(json.dumps({"type": "error", "msg": "Invalid params"}))
        await ws.close()
        return

    loop = asyncio.get_event_loop()
    try:
        await ws.send_text(json.dumps({"type": "status", "msg": "Computing ground truth..."}))
        gt = await loop.run_in_executor(
            None, _gt_energy_series, theta1, theta2, omega1, omega2, n)
        await ws.send_text(json.dumps({"type": "batch", "model": "actual", **gt}))

        await ws.send_text(json.dumps({"type": "status", "msg": "Computing HNN..."}))
        hnn_e = await loop.run_in_executor(
            None, _hnn_energy_series, theta1, theta2, omega1, omega2, n)
        await ws.send_text(json.dumps({"type": "batch", "model": "hnn", **hnn_e}))

        await ws.send_text(json.dumps({"type": "status", "msg": "Computing LSTM..."}))
        lstm_e = await loop.run_in_executor(
            None, _lstm_energy_series, theta1, theta2, omega1, omega2, n)
        await ws.send_text(json.dumps({"type": "batch", "model": "lstm", **lstm_e}))

        await ws.send_text(json.dumps({"type": "done"}))
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
