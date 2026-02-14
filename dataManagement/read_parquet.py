'''
    This file ...
'''
import polars as pl 
from pathlib import Path
from dataclasses import dataclass

#import numpy as np
#from scipy.integrate import odeint

filepath = Path("/data.parquet") # <placeholder>
scan = pl.scan_parquet(filepath) # allows read of table segment
data_window = scan.filter(pl.col("t") < 1).collect() #<check time complexity>

def batch_stream(filepath, t_start, t_end):
  

# for testing
sample_df = pl.DataFrame({ # <placeholder>
    "t":[0.],
    "theta1":[0.],
    "theta2":[0.],
    "l1":[1.],
    "l2":[1.],
    "m1":[1.],
    "m2":[1.],
    "dt":[1e-5],
    "sim_id":["sim00001"],
})

@dataclass
class State:
  # dynamic
  float: theta1
  float: theta2
  #float: w1
  #float: w2
  float: t
  # sim constants
  float: l1
  float: l2
  float: m1
  float: m2
  float: dt
  str: sim_id