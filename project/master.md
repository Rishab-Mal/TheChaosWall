# Master Project Tracker — Double Pendulum LSTM

Goal: Train an LSTM to predict double pendulum dynamics from simulation data stored in Parquet.

---

## Architecture Overview

```
src/                        ← Physics simulation + data generation
  pendulum.py               ← Double pendulum ODE model
  main.py                   ← Simulation sweep + Parquet write driver
  data/
    write_parquet.py        ← SafeBufferedParquetWriter (PyArrow)
    read_parquet.py         ← Polars read, filter, batch stream, (X, y) pairs

project/                    ← ML pipeline
  data/utils.py             ← sim_dict index builder + sample_subsequence (stub)
  models/
    BaseRNN.py              ← BaseRNN class skeleton
    layers.py               ← RNN layer factory
    utils.py                ← load_from_parquet stub (no-op)
    init.py                 ← WRONG FILENAME — must be __init__.py
  training/
    train.py                ← train loop, validate, checkpoint save/load
```

---

## What Is Done

### Physics / Data Generation
- [x] `DoublePendulum` class — full ODE equations, scipy RK45 integration, returns DataFrame
- [x] `generateTimeData()` — produces `[t, theta1, theta2, theta1_dot, theta2_dot]` per sim
- [x] `velocity_verlet_step()` — symplectic integrator implemented and working
- [x] `src/main.py` — sweeps N random ICs, writes each sim to Parquet via SafeBufferedParquetWriter

### Data Storage
- [x] `SafeBufferedParquetWriter` — auto-flush at row_group_size, ZSTD compression, overwrite protection
- [x] `get_pendulum_schema()` — 11-column Arrow schema (`sim_id, t, theta1, theta2, theta1_dot, theta2_dot, l1, l2, m1, m2, dt`)

### Data Reading
- [x] `get_lazy_scan()` — lazy Polars scan, no RAM load
- [x] `load_time_window()` — filter by time range with predicate pushdown
- [x] `load_single_simulation()` — filter by sim_id
- [x] `batch_stream()` — streaming batch iterator (good for large files)
- [x] `generate_training_pairs()` — yields (X, y) shifted pairs for 1-step prediction
- [x] `load_sims_from_parquet()` — builds `{sim_id: (t_0, t_end)}` index dict

### Model
- [x] `BaseRNN` class skeleton — wraps LSTM or GRU via `build_rnn()`
- [x] `init_hidden()` — correctly returns `(h_0, c_0)` tuple for LSTM, single tensor for GRU
- [x] `layers.py` — `stacked_rnn_layers()` factory supporting LSTM / GRU / plain RNN

### Training Infrastructure
- [x] `save_checkpoint()` — saves epoch, model weights, optimizer state
- [x] `load_checkpoint()` — restores all three correctly
- [x] `train_one_epoch()` — skeleton structure (broken, see below)
- [x] `validate()` — skeleton structure (broken, see below)

---

## What Is Broken (Crashes at Runtime)

### project/data/utils.py
- [ ] `.groupby()` → renamed to `.group_by()` in Polars >= 0.19 — AttributeError at runtime
- [ ] `random` is imported but never used — dead import

### project/models/BaseRNN.py
- [ ] `forward()` returns raw tuple `(output, hidden)` from the RNN — `MSELoss(pred, target)` will crash because pred is not a tensor
- [ ] No `output_size` parameter in `__init__` — the `__main__` block passes `output_size=N_FEATURES` and gets a TypeError
- [ ] No output projection head — missing `self.fc = nn.Linear(hidden_size, output_size)` and the `return self.fc(output[:, -1, :])` in forward
- [ ] `__main__` also omits `rnn_type` — second TypeError
- [ ] `init_hidden()` creates tensors on CPU always — device mismatch if model is on GPU
- [ ] `DataLoader` / `TensorDataset` imported at the top of a model file — wrong place

### project/models/utils.py
- [ ] `load_from_parquet()` calls `pl.scan_parquet()` but discards the result and returns nothing — complete no-op

### project/models/init.py
- [ ] Filename is `init.py` — Python will never treat this as a package init. Must be renamed to `__init__.py`

### project/training/train.py
- [ ] `criterion` used inside `train_one_epoch` but not passed as a parameter and not in scope — NameError
- [ ] `model.device` does not exist on `nn.Module` — AttributeError. Use `next(model.parameters()).device` or pass device as a parameter
- [ ] `load_dataset` called in `train_model` but never imported or defined — NameError
- [ ] `MyModel` referenced in `train_model` but never imported — NameError
- [ ] `CrossEntropyLoss` is for classification — this is a regression task, use `MSELoss`
- [ ] `validate()` accesses `dataloader.val_data` and `dataloader.val_targets` — DataLoader has no such attributes. Must iterate over the dataloader like `train_one_epoch` does

### src/pendulum.py
- [ ] `velocity_verlet_step2()` — stub, missing the integration loop and return statement

---

## What Is Not Done Yet

### Data Pipeline
- [ ] `sample_subsequence()` body — needs to pick random (sim_id, t_start), filter the window, return a tensor
- [ ] `Pipeline` class — wraps a file path and exposes `get_batch()` / `__iter__` so the training loop can pull batches without preloading everything
- [ ] Input normalization / standardization — angles and velocities need to be scaled before feeding the LSTM
- [ ] Sliding window construction — current `generate_training_pairs()` does 1-step prediction; LSTM needs sequences of length W as input, not single states
- [ ] Cross-simulation boundary fix in `generate_training_pairs()` — `shift(-1)` across batch boundaries incorrectly pairs the last timestep of one sim with the first timestep of the next sim; need to group by `sim_id` before shifting

### Model
- [ ] Output projection head (`nn.Linear(hidden_size, output_size)`) in `BaseRNN`
- [ ] Consolidate `build_rnn()` (BaseRNN.py) and `stacked_rnn_layers()` (layers.py) — they do the same thing; pick one and delete the other
- [ ] Proper device handling throughout — model, tensors, and hidden state must all land on the same device
- [ ] `models/utils.py` — decide if this file is needed at all; its function is a no-op and duplicates `data/utils.py`

### Training Loop
- [ ] Fix `train_one_epoch` — add `criterion` and `device` as parameters, remove `model.device`
- [ ] Fix `validate()` — replace direct attribute access with a proper batch loop and average the loss
- [ ] Replace `MyModel` / `load_dataset` stubs with actual `BaseRNN` and the Polars data pipeline
- [ ] Add device setup: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- [ ] Add learning rate scheduler (e.g. `ReduceLROnPlateau` on val loss)
- [ ] Add early stopping (stop if val loss hasn't improved for N epochs)
- [ ] Add checkpoint saving on best val loss, not just every epoch
- [ ] Add a `__main__` block / config dict to run training end-to-end from one entry point
- [ ] Connect data pipeline: feed `generate_training_pairs()` or the `Pipeline` class into `DataLoader`

### Evaluation
- [ ] Per-dimension error tracking (theta1 error, theta2 error separately)
- [ ] Multi-step rollout error — predict N steps ahead by feeding model's own output back in
- [ ] Lyapunov exponent comparison — check if the predicted trajectory diverges at the right rate
- [ ] Visualization of predicted vs true trajectories

### Infrastructure
- [ ] Logging (TensorBoard or W&B, or at minimum a loss CSV)
- [ ] SLURM job scripts for running data generation and training on HiPerGator
- [ ] `requirements.txt` aligned between `src/` and `project/`
- [ ] Test suite (at least smoke tests for the data pipeline and model forward pass)
