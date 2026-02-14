##//src/data/write_parquet.py
#import os
#import pyarrow as pa
#import pyarrow.parquet as pq
#
#class SafeBufferedParquetWriter:
#    """
#    Buffered Parquet writer:
#    - Only writes to a new file (fails if file exists)
#    - insert_data(row) stores rows in RAM buffer
#    - flush_buffer() writes the buffer as a row group to disk
#    - Memory usage is minimal (only the buffer is in RAM)
#    """
#
#    def __init__(self, file_name, column_names, row_group_size=1024):
#        self.file_name = file_name
#        self.column_names = column_names
#        self.schema = pa.schema([pa.field(c, pa.float64()) for c in column_names])
#        self.row_group_size = row_group_size
#        self.buffer = []
#
#        # Fail if file already exists
#        if os.path.exists(file_name):
#            raise FileExistsError(f"Parquet file '{file_name}' already exists. Cannot overwrite.")
#
#        # Ensure folder exists
#        os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)
#
#        # Open ParquetWriter
#        self.writer = pq.ParquetWriter(file_name, self.schema, compression="zstd", use_dictionary=True)
#        print(f"Opened ParquetWriter for new file: {file_name}")
#
#    def insert_data(self, row):
#        """
#        Add a single row to the buffer.
#        """
#        if len(row) != len(self.column_names):
#            raise ValueError(f"Row length {len(row)} does not match number of columns {len(self.column_names)}")
#        self.buffer.append(row)
#
#    def flush_buffer(self):
#        """
#        Write buffered rows as a row group to the Parquet file and clear the buffer.
#        """
#        if not self.buffer:
#            print("Buffer is empty, nothing to flush.")
#            return
#
#        # Convert buffer to column-wise dict
#        data_dict = {col: [row[i] for row in self.buffer] for i, col in enumerate(self.column_names)}
#        table = pa.Table.from_pydict(data_dict, schema=self.schema)
#
#        # Write table as a new row group
#        self.writer.write_table(table)
#        print(f"Flushed {len(self.buffer)} rows to Parquet file")
#
#        # Clear buffer
#        self.buffer = []
#
#    def close(self):
#        """
#        Flush any remaining rows and close the writer.
#        """
#        self.flush_buffer()
#        self.writer.close()
#        print(f"Parquet file closed: {self.file_name}")
#
#
## ----------------------------
## Example usage
## ----------------------------
#if __name__ == "__main__":
#    columns = ["id", "value1", "value2", "value3", "value4", "value5", "value6", "value7"]
#
#    # Attempt to create a new Parquet file
#    try:
#        writer = SafeBufferedParquetWriter("buffered_data.parquet", columns)
#    except FileExistsError as e:
#        print(e)
#        exit(1)
#
#    # Push some rows into the buffer
#    for i in range(10):
#        row = [i] + [0.1*j for j in range(7)]
#        writer.insert_data(row)
#
#    # Flush first batch manually
#    writer.flush_buffer()
#
#    # Push more rows
#    for i in range(10, 15):
#        row = [i] + [0.2*j for j in range(7)]
#        writer.insert_data(row)
#
#    # Final flush and close
#    writer.close()
#

# //src/data/write_parquet.py

import os
import pyarrow as pa
import pyarrow.parquet as pq


class SafeBufferedParquetWriter:
    """
    Memory-safe buffered Parquet writer.

    Features:
    - Fails if file already exists (no accidental overwrite)
    - Auto-flush when buffer reaches row_group_size
    - Supports mixed column types (float, string, int)
    - ZSTD compression (good balance for local + HPC)
    """

    def __init__(
        self,
        file_name: str,
        schema: pa.Schema,
        row_group_size: int = 50_000,   # good local default
    ):
        self.file_name = file_name
        self.schema = schema
        self.row_group_size = row_group_size
        self.buffer = []

        # Safety: don't overwrite existing file
        if os.path.exists(file_name):
            raise FileExistsError(
                f"Parquet file '{file_name}' already exists. Refusing to overwrite."
            )

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)

        # Open writer
        self.writer = pq.ParquetWriter(
            file_name,
            self.schema,
            compression="zstd",
            use_dictionary=True,
        )

        print(f"[ParquetWriter] Created new file: {file_name}")

    # ---------------------------------
    # Insert row
    # ---------------------------------
    def insert_data(self, row: tuple):
        """
        Insert a single row (tuple matching schema).
        Auto-flush if buffer reaches row_group_size.
        """
        if len(row) != len(self.schema):
            raise ValueError(
                f"Row length {len(row)} does not match schema length {len(self.schema)}"
            )

        self.buffer.append(row)

        if len(self.buffer) >= self.row_group_size:
            self.flush_buffer()

    # ---------------------------------
    # Flush buffer
    # ---------------------------------
    def flush_buffer(self):
        """
        Write buffered rows as a row group.
        Keeps memory usage controlled.
        """
        if not self.buffer:
            return

        # Convert row-wise buffer → column-wise dict
        columns = {field.name: [] for field in self.schema}

        for row in self.buffer:
            for i, field in enumerate(self.schema):
                columns[field.name].append(row[i])

        table = pa.Table.from_pydict(columns, schema=self.schema)

        self.writer.write_table(table)

        print(f"[ParquetWriter] Flushed {len(self.buffer)} rows")

        self.buffer = []

    # ---------------------------------
    # Close writer
    # ---------------------------------
    def close(self):
        """
        Flush remaining rows and close file.
        """
        if self.buffer:
            self.flush_buffer()

        self.writer.close()
        print(f"[ParquetWriter] Closed file: {self.file_name}")


# ---------------------------------
# Suggested Schema For Your Project
# ---------------------------------

def get_pendulum_schema():
    """
    Returns schema for double pendulum ML dataset.
    """

    return pa.schema([
        pa.field("sim_id", pa.string()),
        pa.field("t", pa.float64()),
        pa.field("theta1", pa.float64()),
        pa.field("theta2", pa.float64()),
        pa.field("theta1_dot", pa.float64()),
        pa.field("theta2_dot", pa.float64()),
        pa.field("l1", pa.float64()),
        pa.field("l2", pa.float64()),
        pa.field("m1", pa.float64()),
        pa.field("m2", pa.float64()),
        pa.field("dt", pa.float64()),
    ])


# ---------------------------------
# Example Usage
# ---------------------------------

if __name__ == "__main__":

    file_name = "pendulum_data.parquet"
    schema = get_pendulum_schema()

    try:
        writer = SafeBufferedParquetWriter(
            file_name=file_name,
            schema=schema,
            row_group_size=10_000  # smaller for local testing
        )
    except FileExistsError as e:
        print(e)
        exit(1)

    # Example small simulation insert
    for i in range(25_000):
        row = (
            "sim00001",  # sim_id
            i * 0.01,    # t
            0.5,         # theta1
            1.0,         # theta2
            0.0,         # theta1_dot
            0.0,         # theta2_dot
            1.0,         # l1
            1.0,         # l2
            1.0,         # m1
            1.0,         # m2
            0.01         # dt
        )
        writer.insert_data(row)

    writer.close()
