import torch


def _get_device() -> str:
    # DirectML doesn't support LSTM or torch.autograd.grad (needed by HNN).
    # Use CUDA if available, otherwise CPU.
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


INPUT_DIM  = 4        # [theta1, theta2, theta1_dot, theta2_dot]
HIDDEN_DIM = 128
LR         = 1e-3
EPOCHS     = 200
SEQ_LEN    = 20
BATCH_SIZE = 64
RNN_HIDDEN = 64
RNN_PATH     = "models/rnn.pth"
PARQUET_PATH = "training_data.parquet"
DEVICE = _get_device()
