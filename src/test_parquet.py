from pendulum import DoublePendulum
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

def main():
    parquet_file = "test_pendulum.parquet"
    writer = None
    count = 0

    # SMALL TEST GRID
    for init_theta1 in np.arange(0, 10, 5):      # 0, 5 degrees
        for init_theta2 in np.arange(0, 20, 10): # 0, 10 degrees
            
            pendulum = DoublePendulum(init_theta1, init_theta2)
            df = pendulum.generateTimeData()

            table = pa.Table.from_pandas(df)

            if writer is None:
                writer = pq.ParquetWriter(parquet_file, table.schema)

            writer.write_table(table)
            count += 1

            print(f"Simulation {count} complete")

    if writer:
        writer.close()

    print(f"\nSaved {count} simulations to {parquet_file}")

if __name__ == "__main__":
    main()
