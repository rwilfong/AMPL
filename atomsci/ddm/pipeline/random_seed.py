""" Used to set random seed from parameter_parser for reproducibility. """
from atomsci.ddm.pipeline import parameter_parser as parse
import pandas as pd 
import numpy as np 
import uuid 
import random
import torch
import tensorflow as tf
#----------------------------------------------------------------------------------
class RandomStateGenerator:
    """
    A class to manage random state and seed generation for reproducible randomness.

    Attributes:
        params: Additional parameters.
        seed: The seed for the random state.
        random_state: The random state generator.
    """
    def __init__(self, params=None, seed=None):
        self.params = params
        if seed is not None:
            self.seed = seed
        else:
            self.seed = uuid.uuid4().int % (2**32)
        self.set_seed(self.seed)
    
    def set_seed(self, seed):
        """Set the seed for all relevant libraries."""
        self.seed = seed
        global _seed, _random_state
        _seed = seed
        _random_state = np.random.default_rng(_seed)
        
        # Set seed for numpy
        np.random.default_rng(_seed)
        # needed for deepchem I think 
        np.random.seed(_seed)
        
        # Set seed for random
        random.seed(_seed)
        
        # Set seed for PyTorch
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # set seed for tensorflow 
        tf.random.set_seed(_seed)
        
        self.random_state = _random_state

    def get_seed(self):
        return self.seed
    
    def get_random_state(self):
        return self.random_state
