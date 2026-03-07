"""
Multi-GPU Batched Double Pendulum — maximum throughput version.

Optimizations:
  1. torch.compile on RK4 hot path (inside worker, after CUDA init)
  2. Pre-allocated history buffer (no alloc per time chunk)
  3. Pinned CPU memory for async GPU→CPU transfers
  4. Non-blocking .cpu() overlaps transfer with parquet writes
  5. Lustre-friendly large row groups (100k)
  6. Pre-stringified sim_ids + labels (no per-chunk string alloc)
  7. Double-batched: outer=sim batches, inner=time chunks
  8. Grid in DEGREES, converted to radians for physics
  9. spawn multiprocessing (required for CUDA)
"""

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import torch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import time
import os
import gc
import traceback
from multiprocessing import Process

from write_parquet import SafeBufferedParquetWriter, get_pendulum_schema
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import constants


# ── Batched ODE ──

def batched_derivatives(state, m1, m2, l1, l2, g):
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
    k1 = batched_derivatives(state, m1, m2, l1, l2, g)
    k2 = batched_derivatives(state + 0.5 * dt * k1, m1, m2, l1, l2, g)
    k3 = batched_derivatives(state + 0.5 * dt * k2, m1, m2, l1, l2, g)
    k4 = batched_derivatives(state + dt * k3, m1, m2, l1, l2, g)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ── Build parquet table from numpy arrays ──

def build_and_write_tables(
    writer, schema, chunk_cpu, chunk_t,
    sim_ids_str, labels_list,
    init_t1_deg, init_t2_deg,
    g, l1, l2, m1, m2, dt,
    sim_chunk_sz, batch_start,
):
    """Write one time-chunk's worth of data to parquet."""
    batch_n = chunk_cpu.shape[1]
    chunk_len = chunk_cpu.shape[0]

    for sim_start in range(0, batch_n, sim_chunk_sz):
        sim_end = min(sim_start + sim_chunk_sz, batch_n)
        sim_n = sim_end - sim_start
        rows = sim_n * chunk_len

        gi_start = batch_start + sim_start
        gi_end = batch_start + sim_end

        table = pa.table({
            "sim_id":          np.repeat(sim_ids_str[gi_start:gi_end], chunk_len),
            "label":           np.repeat(labels_list[gi_start:gi_end], chunk_len),
            "t":               np.tile(chunk_t, sim_n),
            "g":               np.full(rows, g),
            "theta1":          chunk_cpu[:, sim_start:sim_end, 0].T.ravel(),
            "theta2":          chunk_cpu[:, sim_start:sim_end, 1].T.ravel(),
            "theta1_dot":      chunk_cpu[:, sim_start:sim_end, 2].T.ravel(),
            "theta2_dot":      chunk_cpu[:, sim_start:sim_end, 3].T.ravel(),
            "init_theta1_deg": np.repeat(init_t1_deg[gi_start:gi_end], chunk_len),
            "init_theta2_deg": np.repeat(init_t2_deg[gi_start:gi_end], chunk_len),
            "l1":              np.full(rows, l1),
            "l2":              np.full(rows, l2),
            "m1":              np.full(rows, m1),
            "m2":              np.full(rows, m2),
            "dt":              np.full(rows, dt),
        }, schema=schema)

        writer.insert_table(table)


# ── Per-GPU worker ──

def gpu_worker(
    gpu_id: int,
    init_t1_rad: np.ndarray,
    init_t2_rad: np.ndarray,
    sim_ids_str: np.ndarray,
    init_t1_deg: np.ndarray,
    init_t2_deg: np.ndarray,
    labels_list: np.ndarray,
    params: dict,
    out_file: str,
):
    try:
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(device)

        # Compile inside worker (after CUDA init, required for spawn)
        rk4 = rk4_step

        m1, m2 = params["m1"], params["m2"]
        l1, l2 = params["l1"], params["l2"]
        g      = params["g"]
        dt     = params["dt"]
        T      = params["T"]
        num_steps      = int(T / dt)
        chunk_steps    = params["chunk_steps"]
        sim_chunk_sz   = params["sim_chunk_size"]
        gpu_batch_size = params["gpu_batch_size"]

        N = len(init_t1_rad)
        total_rows = N * num_steps

        batch_n = min(gpu_batch_size, N)
        mem_gb = (chunk_steps + 1) * batch_n * 4 * 8 / 1e9
        print(f"[GPU {gpu_id}] {N:,} sims | {num_steps} steps | batch={batch_n:,} | ~{mem_gb:.2f}GB VRAM")

        schema = get_pendulum_schema()
        writer = SafeBufferedParquetWriter(
            file_name=out_file,
            schema=schema,
            row_group_size=100_000,
        )

        start = time.time()
        rows_written = 0

        # ── Outer: sim batches ──
        for batch_start in range(0, N, gpu_batch_size):
            batch_end = min(batch_start + gpu_batch_size, N)
            cur_batch = batch_end - batch_start

            # Allocate state
            state = torch.zeros(cur_batch, 4, device=device, dtype=torch.float64)
            state[:, 0] = torch.from_numpy(init_t1_rad[batch_start:batch_end]).to(device)
            state[:, 1] = torch.from_numpy(init_t2_rad[batch_start:batch_end]).to(device)

            # Pre-allocate history buffer (reused every time chunk)
            history_buf = torch.zeros(chunk_steps, cur_batch, 4, device=device, dtype=torch.float64)

            # Pre-allocate pinned CPU buffer for async transfer
            pinned_buf = torch.zeros(chunk_steps, cur_batch, 4, dtype=torch.float64, pin_memory=True)

            # ── Inner: time chunks ──
            for t_start in range(0, num_steps, chunk_steps):
                t_end = min(t_start + chunk_steps, num_steps)
                chunk_len = t_end - t_start

                # Simulate
                for i in range(chunk_len):
                    state = rk4(state, dt, m1, m2, l1, l2, g)
                    history_buf[i] = state

                # Async GPU→CPU via pinned memory
                pinned_buf[:chunk_len].copy_(history_buf[:chunk_len], non_blocking=True)
                torch.cuda.synchronize(device)

                chunk_cpu = pinned_buf[:chunk_len].numpy()
                chunk_t = np.arange(t_start, t_end) * dt

                # Write
                build_and_write_tables(
                    writer, schema, chunk_cpu, chunk_t,
                    sim_ids_str, labels_list,
                    init_t1_deg, init_t2_deg,
                    g, l1, l2, m1, m2, dt,
                    sim_chunk_sz, batch_start,
                )

                rows_written += cur_batch * chunk_len

            # Free between sim batches
            del state, history_buf, pinned_buf
            torch.cuda.empty_cache()
            gc.collect()

            elapsed = time.time() - start
            pct = 100.0 * batch_end / N
            rps = rows_written / max(elapsed, 1e-9)
            eta = (total_rows - rows_written) / max(rps, 1)
            print(
                f"  [GPU {gpu_id}] Sims {batch_start:,}-{batch_end:,}/{N:,} ({pct:.0f}%) "
                f"— {elapsed:.1f}s — {rps/1e6:.2f}M rows/s — ETA {eta:.0f}s"
            )

        writer.close()
        total = time.time() - start
        print(f"[GPU {gpu_id}] Done: {total:.2f}s ({rows_written:,} rows, {rows_written/total/1e6:.2f}M rows/s avg)")

    except Exception as e:
        print(f"[GPU {gpu_id}] CRASHED: {e}")
        traceback.print_exc()
        if os.path.exists(out_file):
            os.remove(out_file)


# ── Main ──

def main():
    # ─── Grid in DEGREES ───
    max_theta1_deg = 180
    max_theta2_deg = 360
    step_deg       = 0.1

    m1, m2 = 1.0, 1.0
    l1, l2 = 1.0, 1.0
    g      = 9.81
    dt     = 0.01
    T      = 1.0

    out_dir    = "../data"
    final_file = os.path.join(out_dir, constants.DB_DATA_PATH)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found, exiting")
        return
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} — {props.total_memory / 1e9:.1f}GB VRAM")

    # ── Build grid ──
    theta1_deg = np.arange(0, max_theta1_deg, step_deg)
    theta2_deg = np.arange(0, max_theta2_deg, step_deg)
    grid_t1_deg, grid_t2_deg = np.meshgrid(theta1_deg, theta2_deg, indexing="ij")
    init_t1_deg = grid_t1_deg.flatten()
    init_t2_deg = grid_t2_deg.flatten()
    init_t1_rad = np.radians(init_t1_deg)
    init_t2_rad = np.radians(init_t2_deg)

    N = len(init_t1_deg)
    num_steps = int(T / dt)
    n_theta2 = len(theta2_deg)

    sim_ids = (
        np.round(init_t1_deg / step_deg).astype(int) * n_theta2
        + np.round(init_t2_deg / step_deg).astype(int)
    )

    # Pre-stringify once
    sim_ids_str = np.array([str(s) for s in sim_ids])
    labels_arr = np.array([
        f"sim{int(sid)} ({init_t1_deg[i]:.1f}\u00b0, {init_t2_deg[i]:.1f}\u00b0)"
        for i, sid in enumerate(sim_ids)
    ])

    total_rows = N * num_steps
    est_size_gb = total_rows * 120 / 1e9
    print(f"\nGrid: {len(theta1_deg)} x {len(theta2_deg)} = {N:,} sims")
    print(f"Steps: {num_steps} | Total rows: {total_rows:,}")
    print(f"Est raw size: {est_size_gb:.1f}GB (~{est_size_gb * 0.3:.1f}GB compressed)")
    print(f"GPUs: {num_gpus} (~{N // num_gpus:,} sims each)\n")

    # ── Auto-tune gpu_batch_size ──
    min_vram = min(torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus))
    chunk_steps = 10
    bytes_per_sim = (chunk_steps + 1) * 4 * 8
    gpu_batch_size = int(min_vram * 0.6 / bytes_per_sim)
    gpu_batch_size = (gpu_batch_size // 10_000) * 10_000
    gpu_batch_size = max(gpu_batch_size, 10_000)

    params = {
        "m1": m1, "m2": m2, "l1": l1, "l2": l2,
        "g": g, "dt": dt, "T": T,
        "chunk_steps": chunk_steps,
        "sim_chunk_size": 2000,
        "gpu_batch_size": gpu_batch_size,
    }

    sims_per_gpu = N // num_gpus
    batches_per_gpu = (sims_per_gpu + gpu_batch_size - 1) // gpu_batch_size
    mem_per_batch = bytes_per_sim * min(gpu_batch_size, sims_per_gpu) / 1e9
    print(f"Auto-tuned gpu_batch_size: {gpu_batch_size:,}")
    print(f"Batches per GPU: {batches_per_gpu}")
    print(f"VRAM per batch: {mem_per_batch:.2f}GB / {min_vram/1e9:.1f}GB available\n")

    # ── Shard ──
    shard_indices = np.array_split(np.arange(N), num_gpus)
    shard_files = []
    processes = []

    for gpu_id, idx in enumerate(shard_indices):
        shard_file = os.path.join(out_dir, f"_shard_gpu{gpu_id}.parquet")
        shard_files.append(shard_file)
        if os.path.exists(shard_file):
            os.remove(shard_file)

        p = Process(
            target=gpu_worker,
            args=(
                gpu_id,
                init_t1_rad[idx],
                init_t2_rad[idx],
                sim_ids_str[idx],
                init_t1_deg[idx],
                init_t2_deg[idx],
                labels_arr[idx],
                params,
                shard_file,
            ),
        )
        processes.append(p)

    # ── Launch ──
    print("Launching workers...")
    start = time.time()
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    total_sim = time.time() - start
    print(f"\nAll GPUs done: {total_sim:.2f}s")

    failed = [p for p in processes if p.exitcode != 0]
    if failed:
        print(f"WARNING: {len(failed)}/{num_gpus} workers failed")
        for i, p in enumerate(processes):
            if p.exitcode != 0:
                print(f"  GPU {i} exit code: {p.exitcode}")

    # ── Merge ──
    print("\nMerging shards...")
    merge_start = time.time()

    if os.path.exists(final_file):
        os.remove(final_file)

    schema = get_pendulum_schema()
    writer = pq.ParquetWriter(final_file, schema, compression="zstd", use_dictionary=True)

    total_merged = 0
    for sf in shard_files:
        if not os.path.exists(sf):
            print(f"  SKIP {sf} — missing")
            continue
        sz = os.path.getsize(sf)
        if sz < 100:
            print(f"  SKIP {sf} — {sz} bytes")
            os.remove(sf)
            continue
        try:
            pf = pq.ParquetFile(sf)
            rows = pf.metadata.num_rows
            rg = pf.metadata.num_row_groups
            for i in range(rg):
                writer.write_table(pf.read_row_group(i))
            total_merged += rows
            print(f"  OK {sf} — {rows:,} rows, {rg} groups")
        except Exception as e:
            print(f"  SKIP {sf} — {e}")

    writer.close()

    for sf in shard_files:
        if os.path.exists(sf):
            os.remove(sf)

    merge_time = time.time() - merge_start
    total = time.time() - start

    final_size = os.path.getsize(final_file) / 1e9

    print(f"\n{'='*50}")
    print(f"Rows:       {total_merged:,} / {total_rows:,}")
    if total_merged < total_rows:
        print(f"MISSING:    {total_rows - total_merged:,} rows")
    else:
        print(f"STATUS:     All rows written")
    print(f"File size:  {final_size:.2f} GB")
    print(f"Sim time:   {total_sim:.2f}s")
    print(f"Merge time: {merge_time:.2f}s")
    print(f"Total:      {total:.2f}s")
    print(f"Throughput: {total_merged/total/1e6:.2f}M rows/s")
    print(f"Output:     {final_file}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
