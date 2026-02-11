import os
import pyarrow as pa
import pyarrow.parquet as pq

class SafeBufferedParquetWriter:
    """
    Buffered Parquet writer:
    - Only writes to a new file (fails if file exists)
    - insert_data(row) stores rows in RAM buffer
    - flush_buffer() writes the buffer as a row group to disk
    - Memory usage is minimal (only the buffer is in RAM)
    """

    def __init__(self, file_name, column_names, row_group_size=1024):
        self.file_name = file_name
        self.column_names = column_names
        self.schema = pa.schema([pa.field(c, pa.float64()) for c in column_names])
        self.row_group_size = row_group_size
        self.buffer = []

        # Fail if file already exists
        if os.path.exists(file_name):
            raise FileExistsError(f"Parquet file '{file_name}' already exists. Cannot overwrite.")

        # Ensure folder exists
        os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)

        # Open ParquetWriter
        self.writer = pq.ParquetWriter(file_name, self.schema, compression="zstd", use_dictionary=True)
        print(f"Opened ParquetWriter for new file: {file_name}")

    def insert_data(self, row):
        """
        Add a single row to the buffer.
        """
        if len(row) != len(self.column_names):
            raise ValueError(f"Row length {len(row)} does not match number of columns {len(self.column_names)}")
        self.buffer.append(row)

    def flush_buffer(self):
        """
        Write buffered rows as a row group to the Parquet file and clear the buffer.
        """
        if not self.buffer:
            print("Buffer is empty, nothing to flush.")
            return

        # Convert buffer to column-wise dict
        data_dict = {col: [row[i] for row in self.buffer] for i, col in enumerate(self.column_names)}
        table = pa.Table.from_pydict(data_dict, schema=self.schema)

        # Write table as a new row group
        self.writer.write_table(table)
        print(f"Flushed {len(self.buffer)} rows to Parquet file")

        # Clear buffer
        self.buffer = []

    def close(self):
        """
        Flush any remaining rows and close the writer.
        """
        self.flush_buffer()
        self.writer.close()
        print(f"Parquet file closed: {self.file_name}")


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    columns = ["id", "value1", "value2", "value3", "value4", "value5", "value6", "value7"]

    # Attempt to create a new Parquet file
    try:
        writer = SafeBufferedParquetWriter("buffered_data.parquet", columns)
    except FileExistsError as e:
        print(e)
        exit(1)

    # Push some rows into the buffer
    for i in range(10):
        row = [i] + [0.1*j for j in range(7)]
        writer.insert_data(row)

    # Flush first batch manually
    writer.flush_buffer()

    # Push more rows
    for i in range(10, 15):
        row = [i] + [0.2*j for j in range(7)]
        writer.insert_data(row)

    # Final flush and close
    writer.close()
