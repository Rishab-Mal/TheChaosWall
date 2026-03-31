import sys
import asyncio
import json
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

ACTUAL_PARQUET = root / "comparisons" / "actual.parquet"
RNN_PARQUET    = root / "comparisons" / "rnn_predicted.parquet"
HNN_PARQUET    = root / "comparisons" / "hnn_predicted.parquet"

COLS = ["t", "theta1", "theta2", "theta1_dot", "theta2_dot"]


def load_frames() -> list[dict]:
    actual = pl.read_parquet(str(ACTUAL_PARQUET)).sort("t").select(COLS)
    rnn    = pl.read_parquet(str(RNN_PARQUET)).sort("t").select(COLS)
    hnn    = pl.read_parquet(str(HNN_PARQUET)).sort("t").select(COLS)

    n = min(len(actual), len(rnn), len(hnn))
    frames = []
    for i in range(n):
        ar = actual.row(i, named=True)
        rr = rnn.row(i, named=True)
        hr = hnn.row(i, named=True)
        frames.append({
            "t": round(ar["t"], 4),
            "actual": {
                "theta1": ar["theta1"], "theta2": ar["theta2"],
                "omega1": ar["theta1_dot"], "omega2": ar["theta2_dot"],
            },
            "lstm": {
                "theta1": rr["theta1"], "theta2": rr["theta2"],
                "omega1": rr["theta1_dot"], "omega2": rr["theta2_dot"],
            },
            "hnn": {
                "theta1": hr["theta1"], "theta2": hr["theta2"],
                "omega1": hr["theta1_dot"], "omega2": hr["theta2_dot"],
            },
        })
    return frames


@app.get("/")
async def index():
    return HTMLResponse("<html><body>WebSocket at <code>/ws</code></body></html>")


@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    await ws.accept()

    missing = [
        p.name for p in [ACTUAL_PARQUET, RNN_PARQUET, HNN_PARQUET]
        if not p.exists()
    ]
    if missing:
        await ws.send_text(json.dumps({
            "error": f"Missing files: {', '.join(missing)} — run compare scripts first"
        }))
        await ws.close()
        return

    frames = load_frames()
    try:
        while True:
            for frame in frames:
                await ws.send_text(json.dumps(frame))
                await asyncio.sleep(0.05)   # 20 fps, matches dt=0.05 in training data
    except (WebSocketDisconnect, Exception):
        pass
