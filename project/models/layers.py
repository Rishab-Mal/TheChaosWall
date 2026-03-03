import torch.nn as nn


def build_rnn(rnn_type, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=False):
    """
    Factory: returns an LSTM, GRU, or plain RNN layer.
    All args are passed as kwargs to avoid positional ordering bugs.
    """
    rnn_type = rnn_type.upper()
    common = dict(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        dropout=dropout,
        bidirectional=bidirectional,
    )
    if rnn_type == "LSTM":
        return nn.LSTM(**common)
    elif rnn_type == "GRU":
        return nn.GRU(**common)
    elif rnn_type == "RNN":
        return nn.RNN(**common, nonlinearity='tanh')
    else:
        raise ValueError(f"Unsupported RNN type: '{rnn_type}'. Choose 'LSTM', 'GRU', or 'RNN'.")
