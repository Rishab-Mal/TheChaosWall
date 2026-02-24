from pendulum import DoublePendulum
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

def main():
    parquet_file = "test_pendulum2.parquet"
    writer = None
    num_sims = 0
    max_theta1 = 10
    max_theta2 = 20
    step = 0.1

    # SMALL TEST GRID
<<<<<<< HEAD
    for init_theta1 in np.arange(0, 10, 5):      # 0, 5 degrees
        for init_theta2 in np.arange(0, 20, 10): # 0, 10 degrees
=======
    for init_theta1 in np.arange(0, max_theta1, step):      # 0, 5 degrees
        for init_theta2 in np.arange(0, max_theta2, step):  # 0, 10 degrees
            sim_id = int(round(init_theta1 / step)) * int(max_theta2 / step) + int(round(init_theta2 / step))
>>>>>>> 1271e211a9fa27a6cb93774f6592de7013dca857

            pendulum = DoublePendulum(init_theta1, init_theta2)
            df = pendulum.generateTimeData()
            df["sim_id"] = sim_id
            

            df["run_id"] = count
            df["init_theta1"] = init_theta1
            df["init_theta2"] = init_theta2

            table = pa.Table.from_pandas(df)

            if writer is None:
                writer = pq.ParquetWriter(parquet_file, table.schema)

            writer.write_table(table)
            num_sims += 1

            print(f"Simulation {sim_id} complete")

    if writer:
        writer.close()

    print(f"\nSaved {num_sims} simulations to {parquet_file}")

if __name__ == "__main__":
    main()
