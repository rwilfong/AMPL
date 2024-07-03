""" Used to set random seed from parameter parser for reproducibility. """
from atomsci.ddm.pipeline import parameter_parser as parse
import pandas as pd 
import numpy as np 
import uuid 
# =====================================================================================================
class RandomStateGenerator:
    """
    A class to manage random state and seed generation for reproducible randomness.

    Attributes:
        params:
        seed:

    methods

    """
    def __init__(self, params, seed=None):
        self.params = params
        if seed is not None:
            self.seed = seed
            set_seed(self.seed)
        else:
            self.seed, self.random_state = generate_new_seed()
        self.random_state = get_random_state()
    
    def get_seed(self):
        return self.seed
    
    def get_random_state(self):
        return self.random_state

# Global variables to store seed and random state
_seed = None
_random_state = None

def set_seed(seed):
    global _seed, _random_state
    _seed = seed
    _random_state = np.random.default_rng(_seed)

def generate_new_seed():
    global _seed, _random_state
    _seed = uuid.uuid4().int
    _random_state = np.random.default_rng(_seed)
    return _seed, _random_state

def get_seed():
    return _seed

def get_random_state():
    return _random_state

# --------------------------------------------OLD --------------------------------------------------

#_seed = None 
#_random_state = None 

#def set_seed(seed):
#    global _seed, _random_state
#    _seed = seed 
#    _random_state = np.random.default_rng(_seed)
#
#def generate_new_seed():
#    global _seed, _random_state
#    _seed = uuid.uuid4().int
#    _random_state = np.random.default_rng(_seed)
#    return _seed, _random_state

#def get_seed():
#    return _seed

#def get_random_state():
#    return _random_state

#class RandomStateGenerator:
#    def __init__(self, params, seed=None):
#        self.params = params
#       if seed is not None:
#            self.seed = params.seed
#            set_seed(self.seed)
#        else:
#            self.seed = generate_new_seed()
#        self.random_state = get_random_state() #np.random.default_rng(self.seed)
    
#    def get_seed(self):
#        return self.seed
    
#    def get_random_state(self):
#       return self.random_state
