import pyarrow.parquet as pq
import constants

file_path = constants.DB_DATA_PATH

pf = pq.ParquetFile(file_path)

print("Row groups:", pf.metadata.num_row_groups)
print("Total rows:", pf.metadata.num_rows)
print()

all_sims = set()
prev_count = 0

for rg in range(pf.metadata.num_row_groups):
    # read only the sim_id column
    table = pf.read_row_group(rg, columns=["sim_id"])
    sims = table.column("sim_id").to_pylist()

    all_sims.update(sims)
    curr_count = len(all_sims)

    if curr_count == prev_count:
        print(f"No new simulations added at row group {rg}. Stopping early.")
        break

    prev_count = curr_count

print("\n=======================")
print("TOTAL UNIQUE SIMULATIONS:", len(all_sims))
print("=======================")
