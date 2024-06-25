""" Used to set random seed from parameter parser for reproducibility. """
from atomsci.ddm.pipeline import parameter_parser as parse
import pandas as pd 
import numpy as np 
import uuid 
# =====================================================================================================
def get_random_seed(params):
    """
    Input: params 
    """
    if params.seed is not None:
        seed=params.seed
    else:
        # create a seed 
        seed=uuid.uuid4().int
    rng = np.random.default_rng(seed)
    return seed, rng