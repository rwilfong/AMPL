""" Used to set random seed from parameter parser for reproducibility. """
from atomsci.ddm.pipeline import parameter_parser as parse
import pandas as pd 
import numpy as np 
import uuid 
# =====================================================================================================
_seed = None

def set_seed(seed):
    global _seed
    _seed = seed

def generate_seed():
    global _seed
    if _seed is None:
        _seed = uuid.uuid4().int
    return _seed

def get_seed():
    return _seed

class RandomStateGenerator:
    def __init__(self, params):
        self.params = params
        self.seed = self.params.seed
        if self.params.seed is not None:
            self.seed = self.params.seed
            set_seed(self.seed)
        else:
            self.seed = generate_seed()
            set_seed(self.seed)
        self.random_state = np.random.default_rng(self.seed)
    
    def get_seed(self):
        return self.seed
    
    def get_random_state(self):
        return self.random_state