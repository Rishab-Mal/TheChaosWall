import polars as pl
import random

def load_sims_from_parquet(file_path: str):
    df = pl.scan_parquet(file_path)

    sim_idx = (
        df.select(['sim_id', 't'])
        .group_by('sim_id')
        .agg([
            pl.col('t').max().alias('t_end'),
            pl.col('t').min().alias('t_0'),
            pl.col('t').count().alias('count')
        ])
        .collect()
    )
    sim_dict = {
        row['sim_id']: (row['t_0'], row['t_end'], row['count'])
        for row in sim_idx.iter_rows(named=True)
    }

    return sim_dict

def sample_subsequence(file_path: str, sim_dict: dict, n_samples=20):
    sim_id = random.choice(list(sim_dict.keys()))
    t_0, t_end, count = sim_dict[sim_id]
    
    if isinstance(n_samples, str) and n_samples.endswith('%'):
        pct = float(n_samples[:-1]) / 100
        n_samples = max(1, int(count * pct))

    if count < n_samples:
        raise ValueError(f"Simulation {sim_id} is too short for the requested number of samples.")
        
    start_idx = random.randint(0, count - n_samples)

    query = (
        pl.scan_parquet(file_path)
        .filter(pl.col('sim_id') == sim_id)
        .sort('t')
        .slice(start_idx, n_samples)
    )
    return query.collect()

def sample_batch(file_path: str, sim_dict: dict, batch_size: int=32, n_samples: int=20):
    batch = []
    for _ in range(batch_size):
        subsequence = sample_subsequence(file_path, sim_dict, n_samples)
        batch.append(subsequence)
    return batch

# TODO: split sims into 80/20 train/test sets
# TODO: Create a pipeline class (storing filepath) to create/normalize batches on demand for training