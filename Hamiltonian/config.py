import torch

INPUT_DIM = 4        # [theta1, theta2, theta1_dot, theta2_dot]
HIDDEN_DIM = 128
LR = 1e-3
EPOCHS = 500
SEQ_LEN = 20
BATCH_SIZE = 64
RNN_HIDDEN = 64
RNN_PATH = "models/rnn.pth"
PARQUET_PATH = "test_pendulum2.parquet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
