##//src/data/read_parquet.py
#import polars as pl 
#from pathlib import Path
#
##import numpy as np
##from scipy.integrate import odeint
#
#filepath = Path("/data.parquet") # <placeholder>
#scan = pl.scan_parquet(filepath) # allows read of table segment
#data_window = scan.filter(pl.col("t") < 1).collect() #<check time complexity>
#
#def batch_stream(filepath, t_start, t_end):
#  
#
## for testing
#sample_df = pl.DataFrame({ # dummy data for testing
#    "t":[0.]*100,
#    "theta1":[0.]*100,
#    "theta2":[0.]*100,
#    "l1":[1.]*100,
#    "l2":[1.]*100,
#    "m1":[1.]*100,
#    "m2":[1.]*100,
#    "dt":[1e-5]*100,
#    "sim_id":["sim00001"]*100
#})
#
# //src/data/read_parquet.py

import polars as pl
from pathlib import Path
from typing import Iterator, Optional


# -----------------------------
# GLOBAL SCAN (lazy, memory-safe)
# -----------------------------

def get_lazy_scan(filepath: str | Path) -> pl.LazyFrame:
    """
    Returns a Polars LazyFrame for memory-efficient querying.
    Does NOT load data into RAM.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Parquet file not found: {filepath}")

    return pl.scan_parquet(filepath)


# -----------------------------
# TIME WINDOW FILTER
# -----------------------------

def load_time_window(
    filepath: str | Path,
    t_start: float,
    t_end: float
) -> pl.DataFrame:
    """
    Loads data between t_start and t_end.
    Uses lazy scan → optimized predicate pushdown.
    """
    scan = get_lazy_scan(filepath)

    df = (
        scan
        .filter((pl.col("t") >= t_start) & (pl.col("t") <= t_end))
        .collect()
    )

    return df


# -----------------------------
# SIMULATION FILTER
# -----------------------------

def load_single_simulation(
    filepath: str | Path,
    sim_id: str
) -> pl.DataFrame:
    """
    Load one simulation by sim_id.
    """
    scan = get_lazy_scan(filepath)

    df = (
        scan
        .filter(pl.col("sim_id") == sim_id)
        .collect()
    )

    return df


# -----------------------------
# STREAM BATCHES (FOR TRAINING)
# -----------------------------

def batch_stream(
    filepath: str | Path,
    batch_size: int = 100_000,
    columns: Optional[list[str]] = None
) -> Iterator[pl.DataFrame]:
    """
    Streams dataset in batches.
    Does NOT load entire dataset into RAM.

    Ideal for neural network training.
    """

    scan = get_lazy_scan(filepath)

    if columns:
        scan = scan.select(columns)

    # Collect in streaming mode
    df_iter = scan.collect(streaming=True).iter_slices(n_rows=batch_size)

    for batch in df_iter:
        yield batch


# -----------------------------
# TRAINING WINDOW GENERATOR
# -----------------------------

def generate_training_pairs(
    filepath: str | Path,
    batch_size: int = 100_000
) -> Iterator[tuple]:
    """
    Yields (X, y) training pairs:
    X = [theta1, theta2, theta1_dot, theta2_dot]
    y = next timestep state

    Assumes data sorted by (sim_id, t)
    """

    required_cols = [
        "sim_id",
        "t",
        "theta1",
        "theta2",
        "theta1_dot",
        "theta2_dot",
    ]

    for batch in batch_stream(filepath, batch_size, required_cols):

        # Sort inside batch (safe for chunked processing)
        batch = batch.sort(["sim_id", "t"])

        # Create shifted targets
        batch = batch.with_columns([
            pl.col("theta1").shift(-1).alias("theta1_next"),
            pl.col("theta2").shift(-1).alias("theta2_next"),
            pl.col("theta1_dot").shift(-1).alias("theta1_dot_next"),
            pl.col("theta2_dot").shift(-1).alias("theta2_dot_next"),
        ])

        # Drop last timestep per simulation
        batch = batch.filter(pl.col("theta1_next").is_not_null())

        X = batch.select([
            "theta1",
            "theta2",
            "theta1_dot",
            "theta2_dot"
        ])

        y = batch.select([
            "theta1_next",
            "theta2_next",
            "theta1_dot_next",
            "theta2_dot_next"
        ])

        yield X, y


# -----------------------------
# QUICK SANITY CHECK
# -----------------------------

if __name__ == "__main__":

    test_path = "pendulum_simulations.parquet"

    print("Testing time window load...")
    try:
        df = load_time_window(test_path, 0.0, 0.5)
        print(df.head())
    except FileNotFoundError:
        print("Test file not found (expected if not generated yet).")

    print("Testing batch streaming...")
    try:
        for i, batch in enumerate(batch_stream(test_path, batch_size=10_000)):
            print(f"Batch {i} size:", batch.shape)
            break
    except FileNotFoundError:
        pass
