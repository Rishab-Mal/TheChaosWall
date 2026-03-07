import numpy as np
import pandas as pd
from pathlib import Path

def pendumlum_derivs(state):
    q, p = state
    dqdt = p
    dpdt = -np.sin(q)
    return np.array([dqdt, dpdt])

