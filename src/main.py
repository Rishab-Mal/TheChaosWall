import numpy as np
import constants
from pathlib import Path

from pendulum import DoublePendulum
from data.write_parquet import SafeBufferedParquetWriter, get_pendulum_schema

# Output goes to the project root so train_rnn.py can find it directly.
_ROOT = Path(__file__).resolve().parents[1]


def main():

    # -----------------------------
    # CONFIG
    # -----------------------------
    parquet_file = constants.DB_Path
    num_simulations = 500        # 500 sims × 180 windows = 90,000 training windows
    T  = 10.0                    # 10 s per sim → 200 timesteps (substeps keep h=0.01 internally)
    dt = 0.05
    l1 = 1.0
    l2 = 1.0
    m1 = 1.0
    m2 = 1.0

    print(f"Generating {num_simulations} simulations  (T={T}s, dt={dt}s)")
    print(f"Output: {parquet_file}")
    print(f"Expected windows for training: ~{num_simulations * (int(T/dt) - 20):,}\n")

    # -----------------------------
    # SETUP WRITER
    # -----------------------------
    schema = get_pendulum_schema()

    try:
        writer = SafeBufferedParquetWriter(
            file_name=parquet_file,
            schema=schema,
            row_group_size=50_000  # keeps RAM small
        )
    except FileExistsError as e:
        print(e)
        return

    # -----------------------------
    # GENERATE DATA
    # -----------------------------
    for sim_idx in range(num_simulations):

        # Random initial conditions (radians!)
        init_theta1 = np.random.uniform(0, np.pi)
        init_theta2 = np.random.uniform(0, 2 * np.pi)

        pendulum = DoublePendulum(
            theta1=init_theta1,
            theta2=init_theta2,
            l1=l1,
            l2=l2,
            m1=m1,
            m2=m2,
            dt=dt,
            T=T
        )

        df = pendulum.generateTimeData()

        sim_id = f"sim{sim_idx:06d}"

        # Insert rows into buffered writer
        for _, row in df.iterrows():
            writer.insert_data((
                sim_id,
                float(row["t"]),
                float(row["theta1"]),
                float(row["theta2"]),
                float(row["theta1_dot"]),
                float(row["theta2_dot"]),
                l1,
                l2,
                m1,
                m2,
                dt
            ))

        if (sim_idx + 1) % 50 == 0:
            print(f"  {sim_idx + 1}/{num_simulations} simulations done")

    # -----------------------------
    # CLOSE WRITER
    # -----------------------------
    writer.close()

    print(f"\nFinished. Dataset saved to {parquet_file}")


if __name__ == "__main__":
    main()
