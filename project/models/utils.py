import polars as pl


def load_from_parquet(file_path: str) -> pl.LazyFrame:
    return pl.scan_parquet(file_path)
