##//src/main.py
#from pendulum import DoublePendulum
#import pandas as pd
#import numpy as np
#import pyarrow.parquet as pq
#import pyarrow as pa
#
#def main():
#    parquet_file = 'pendulum_simulations.parquet'
#    writer = None
#    count = 0
#    
#    # NEW PARQUET DOC: columns = init_theta1, init_theta2, t, theta1, theta2, theta1_dot, theta2_dot
#    
#    for init_theta1 in np.arange(0, 180.01, 0.01):  # 0 to 180 degrees
#        for init_theta2 in np.arange(0, 360.01, 0.01):  # 0 to 360 degrees
#            pendulum = DoublePendulum(init_theta1, init_theta2)
#            df = pendulum.generateTimeData()
#            
#            # Convert to pyarrow table
#            table = pa.Table.from_pandas(df)
#            
#            # Initialize writer on first iteration
#            if writer is None:
#                writer = pq.ParquetWriter(parquet_file, table.schema)
#            
#            # Write this simulation's data
#            writer.write_table(table)
#            count += 1
#            
#            if count % 1000 == 0:
#                print(f"Processed {count} simulations...")
#    
#    if writer:
#        writer.close()
#    
#    print(f"Saved {count} simulations to {parquet_file}")
#
#if __name__ == "__main__":
#    main()

# //src/main.py

import numpy as np
from pathlib import Path

from pendulum import DoublePendulum
from data.write_parquet import SafeBufferedParquetWriter, get_pendulum_schema


def main():

    # -----------------------------
    # CONFIG (LOCAL SAFE DEFAULTS)
    # -----------------------------
    parquet_file = "pendulum_simulations.parquet"
    num_simulations = 20_000     # Safe for local machine
    T = 1.0                      # total time
    dt = 0.01                    # timestep
    l1 = 1.0
    l2 = 1.0
    m1 = 1.0
    m2 = 1.0

    print(f"Generating {num_simulations} simulations...")
    print(f"T={T}, dt={dt}")

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

        if (sim_idx + 1) % 1000 == 0:
            print(f"Completed {sim_idx + 1} simulations")

    # -----------------------------
    # CLOSE WRITER
    # -----------------------------
    writer.close()

    print(f"\nFinished. Dataset saved to {parquet_file}")


if __name__ == "__main__":
    main()
