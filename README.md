# Large-Scale Numeric Data Storage & Ingestion (C++ → Parquet)

## Overview
This project generates and stores a very large numeric dataset (**6.4 billion rows × 8 columns**) efficiently using **Apache Arrow and Parquet**.  
The design avoids CSV to reduce storage size, improve performance, and scale cleanly on HPC systems.

---

## Data Characteristics
- **Rows:** 6.4 × 10⁹  
- **Columns:** 8  
- **Type:** Numeric decimals (`float64`)  
- **Raw in-memory size:** ~410 GB  
- **On-disk size (Parquet + ZSTD):** ~40–90 GB  

---

## Why Not CSV?
- Numbers stored as text → **5–10× larger**
- Slow to parse
- No schema enforcement
- Millions of small files stress filesystems

---

## Chosen Format: Apache Parquet
- Columnar binary storage
- Built-in compression (ZSTD)
- Fast analytic reads
- Widely supported (Arrow, Pandas, Spark, DuckDB)

---

## Write Pipeline


---

## Storage Strategy
- Write **few large Parquet files** (100–500 MB each)
- Use row groups of **256k–1M rows**
- One file per process
- Avoid millions of tiny files

---

## Performance Expectations
- **Laptop (SSD):** ~5–12 minutes
- **High-end desktop:** ~2–6 minutes
- **HiPerGator (single node):** < 1 minute
- **HiPerGator (job array / multi-node):** seconds–minutes

---

## HPC Best Practices (HiPerGator)
- Run generation and writes directly on HPC
- Use `$SCRATCH` for large outputs
- Avoid `$HOME` for big data
- Release file handles promptly
- Prefer ZSTD over GZIP

---

## Key Takeaways
- Parquet is **dramatically smaller and faster** than CSV
- Chunked, batched writes are critical for performance
- Data should be generated and stored on HPC, not locally
- This pipeline scales from laptops to HPC clusters with no redesign
