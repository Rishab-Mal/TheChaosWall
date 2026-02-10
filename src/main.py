from pendulum import DoublePendulum
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

def main():
    parquet_file = 'pendulum_simulations.parquet'
    writer = None
    count = 0
    
    # NEW PARQUET DOC: columns = init_theta1, init_theta2, t, theta1, theta2, theta1_dot, theta2_dot
    
    for init_theta1 in np.arange(0, 180.01, 0.01):  # 0 to 180 degrees
        for init_theta2 in np.arange(0, 360.01, 0.01):  # 0 to 360 degrees
            pendulum = DoublePendulum(init_theta1, init_theta2)
            df = pendulum.generateTimeData()
            
            # Convert to pyarrow table
            table = pa.Table.from_pandas(df)
            
            # Initialize writer on first iteration
            if writer is None:
                writer = pq.ParquetWriter(parquet_file, table.schema)
            
            # Write this simulation's data
            writer.write_table(table)
            count += 1
            
            if count % 1000 == 0:
                print(f"Processed {count} simulations...")
    
    if writer:
        writer.close()
    
    print(f"Saved {count} simulations to {parquet_file}")

if __name__ == "__main__":
    main()