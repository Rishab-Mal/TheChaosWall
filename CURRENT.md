Project summary (as of 2026-02-10)

What has been done

- Defined the storage goal: generate massive numeric datasets and store them efficiently in Parquet instead of CSV, with guidance on file sizing, row groups, compression, and HPC usage.
- Implemented a double pendulum simulator that produces time series data per initial angle pair.
- Added a Parquet write pipeline that loops over initial angle grids, runs simulations, and writes each simulation's data to a Parquet file via PyArrow.
- Drafted a safe buffered Parquet writer utility that writes row groups incrementally with ZSTD compression and prevents accidental overwrites.
- Started a Polars-based read path (scanning Parquet and filtering a window) as a placeholder for downstream analysis.

How everything is organized

- README.md: high-level design notes on large-scale numeric data storage and Parquet/HPC strategy.
- PHYSICS.md: placeholder for physics/measurement notes.
- src/pendulum.py: double pendulum model and time integration producing per-step data rows.
- src/main.py: simulation driver that sweeps initial angles and writes Parquet outputs.
- pendulumData.csv: legacy/placeholder CSV data (not the preferred format).
- dataManagement/write_parquet.py: buffered Parquet writer utility (PyArrow).
- dataManagement/read_parquet.py: initial Polars scanning/filtering example (work in progress).
- src/data/: additional data-related scripts and requirements (currently minimal).
- requirements.txt and dataManagement/requirements.txt: Python dependencies (to be aligned/filled as needed).
- Makefile: currently empty.
