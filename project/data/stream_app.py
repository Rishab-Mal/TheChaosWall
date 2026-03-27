import sys
import asyncio
import json
from pathlib import Path
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

# ready models and state dicts
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from Hamiltonian.model import HNN
from Hamiltonian.constants import INPUT_DIM, HIDDEN_DIM, EPOCHS
from project.models import RNNModel

RNN_DICT = root / "models" / "rnn.pth"
HNN_DICT = root / "models" / f"hnn_epoch_{EPOCHS}.pth"
ACTUAL_PARQUET = root / "Comparisons" / "actual.parquet"
RNN_PARQUET = root / "Comparisons" / "rnn_predicted.parquet"
HNN_PARQUET = root / "Comparisons" / "hnn_predicted.parquet"

# load models from pth
rnn_model = RNNModel.RNNModel()
hnn_model = HNN(input_size=INPUT_DIM, hidden_size=HIDDEN_DIM)

async def stream_from_models():
    pass # TODO

# serve frontend
@app.get("/")
async def index():
    return HTMLResponse("<html><body>WebSocket Connected at `/ws`!</body></html>")
    #return HTMLResponse(open("static/index.html").read()) # TODO


@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    if not ACTUAL_PARQUET.exists():
        await ws.send_text(json.dumps({"error":""}))
        # TODO
