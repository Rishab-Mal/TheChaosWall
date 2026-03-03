
import torch
import numpy as np
import pyarrow as pa
import time

from data.write_parquet import SafeBufferedParquetWriter, get_pendulum_schema


# ── Batched ODE (same equations as your DoublePendulum.equations) ──

def batched_derivatives(state, m1, m2, l1, l2, g):
    """
    state: (N, 4) tensor — [theta1, theta2, omega1, omega2]
    Returns: (N, 4) tensor of derivatives
    """
    theta1 = state[:, 0]
    theta2 = state[:, 1]
    omega1 = state[:, 2]
    omega2 = state[:, 3]

    delta = theta1 - theta2
    sin_d = torch.sin(delta)
    cos_d = torch.cos(delta)

    D = (m1 + m2) * l1 - m2 * l1 * cos_d ** 2

    alpha1 = (
        m2 * l1 * omega1 ** 2 * sin_d * cos_d
        + m2 * g * torch.sin(theta2) * cos_d
        + m2 * l2 * omega2 ** 2 * sin_d
        - (m1 + m2) * g * torch.sin(theta1)
    ) / D

    alpha2 = (
        -m2 * l2 * omega2 ** 2 * sin_d * cos_d
        + (m1 + m2) * (
            g * torch.sin(theta1) * cos_d
            - l1 * omega1 ** 2 * sin_d
            - g * torch.sin(theta2)
        )
    ) / (l2 / l1 * D)

    return torch.stack([omega1, omega2, alpha1, alpha2], dim=1)


def rk4_step(state, dt, m1, m2, l1, l2, g):
    """One RK4 step for ALL pendulums simultaneously."""
    k1 = batched_derivatives(state, m1, m2, l1, l2, g)
    k2 = batched_derivatives(state + 0.5 * dt * k1, m1, m2, l1, l2, g)
    k3 = batched_derivatives(state + 0.5 * dt * k2, m1, m2, l1, l2, g)
    k4 = batched_derivatives(state + dt * k3, m1, m2, l1, l2, g)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ── Main ──

def main():
    # Same params as your original
    max_theta1 = 10.0
    max_theta2 = 20.0
    step = 0.1
    m1, m2 = 1.0, 1.0
    l1, l2 = 1.0, 1.0
    g = 9.81
    dt = 0.01
    T = 1.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU detected, falling back to CPU")

    # ── Build initial condition grid ──
    theta1_vals = np.arange(0, max_theta1, step)
    theta2_vals = np.arange(0, max_theta2, step)
    grid_t1, grid_t2 = np.meshgrid(theta1_vals, theta2_vals, indexing="ij")
    init_t1 = grid_t1.flatten()
    init_t2 = grid_t2.flatten()
    N = len(init_t1)
    num_steps = int(T / dt)
    n_theta2 = int(max_theta2 / step)

    print(f"Simulating {N} pendulums x {num_steps} steps on {device}")

    # ── Sim IDs (same scheme as your original) ──
    sim_ids = (
        np.round(init_t1 / step).astype(int) * n_theta2
        + np.round(init_t2 / step).astype(int)
    )

    # ── Initialize state: (N, 4) = [theta1, theta2, 0, 0] ──
    state = torch.zeros(N, 4, device=device, dtype=torch.float64)
    state[:, 0] = torch.from_numpy(init_t1).to(device)
    state[:, 1] = torch.from_numpy(init_t2).to(device)

    # ── Run all timesteps, store history ──
    history = torch.zeros(num_steps, N, 4, device=device, dtype=torch.float64)
    history[0] = state

    start = time.time()
    for i in range(1, num_steps):
        state = rk4_step(state, dt, m1, m2, l1, l2, g)
        history[i] = state

    if device == "cuda":
        torch.cuda.synchronize()
    sim_time = time.time() - start
    print(f"Simulation done: {sim_time:.2f}s")

    # ── Write using bulk insert_table ──
    history_cpu = history.cpu().numpy()  # (num_steps, N, 4)
    t_vals = np.arange(0, T, dt)

    schema = get_pendulum_schema()
    writer = SafeBufferedParquetWriter(
        file_name="test_pendulum.parquet",
        schema=schema,
        row_group_size=50_000,
    )

    print(f"Writing {N * num_steps} rows to parquet...")
    write_start = time.time()

    # Write in chunks of sims to control memory
    chunk_size = 1000
    for chunk_start in range(0, N, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N)
        chunk_n = chunk_end - chunk_start

        # Build columns directly from numpy — no Python loops
        table = pa.table({
            "sim_id":     np.repeat([str(s) for s in sim_ids[chunk_start:chunk_end]], num_steps),
            "t":          np.tile(t_vals, chunk_n),
            "theta1":     history_cpu[:, chunk_start:chunk_end, 0].T.flatten(),
            "theta2":     history_cpu[:, chunk_start:chunk_end, 1].T.flatten(),
            "theta1_dot": history_cpu[:, chunk_start:chunk_end, 2].T.flatten(),
            "theta2_dot": history_cpu[:, chunk_start:chunk_end, 3].T.flatten(),
            "l1":         np.full(chunk_n * num_steps, l1),
            "l2":         np.full(chunk_n * num_steps, l2),
            "m1":         np.full(chunk_n * num_steps, m1),
            "m2":         np.full(chunk_n * num_steps, m2),
            "dt":         np.full(chunk_n * num_steps, dt),
        }, schema=schema)

        writer.insert_table(table)

    writer.close()
    write_time = time.time() - write_start
    total = time.time() - start
    print(f"Write done: {write_time:.2f}s")
    print(f"Total: {total:.2f}s ({N} sims)")


if __name__ == "__main__":
    main()
