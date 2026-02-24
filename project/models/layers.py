import torch.nn as nn

# GOOD: having a dedicated layers.py for reusable building blocks is the right pattern
# WRONG: this function duplicates build_rnn() in BaseRNN.py — two functions doing the same thing in the same project.
#        Consolidate: delete build_rnn() from BaseRNN.py and import from here instead, since this version
#        is more complete (handles plain 'RNN' type which BaseRNN.build_rnn() doesn't).
def stacked_rnn_layers(input_size, hidden_size, num_layers = 1, rnn_type ="LSTM", dropout=0):
    """
    Returns an RNN/GRU/LSTM layer with optional stacking and dropout.
    """
    # GOOD: .upper() makes the type check case-insensitive
    rnn_type = rnn_type.upper()
    if rnn_type == "LSTM":
        # GOOD: batch_first=True is correct for our use case where data is (batch, seq, features)
        rnn_layer = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True, dropout=dropout)
    elif rnn_type == "GRU":
        rnn_layer = nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True, dropout=dropout)
    elif rnn_type == "RNN":
        # GOOD: explicitly setting nonlinearity='tanh' is correct (it's the default but being explicit is fine)
        rnn_layer = nn.RNN(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True, nonlinearity='tanh', dropout=dropout)
    else:
        raise ValueError("Unsupported RNN type: {}".format(rnn_type))
    # WRONG: PyTorch raises a UserWarning (and sometimes errors) when dropout > 0 and num_layers == 1
    #        because dropout only applies between layers. Should guard: if dropout > 0 and num_layers == 1: raise/warn
    return rnn_layer

# NOT DONE: no attention mechanism layer (useful for long sequences)
# NOT DONE: no positional encoding or time-embedding layer for irregular time steps (relevant for physics sims)
# NOT DONE: no output head layer — a nn.Linear(hidden_size, output_size) that sits after the RNN
# NOT DONE: no normalization layers (LayerNorm is commonly applied after RNN output)
