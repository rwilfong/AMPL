"""Unittest to confirm that the sampling implementations from sampling.py actually work before integrating into the pipeline."""
import unittest
from collections import namedtuple
import numpy as np
import deepchem as dc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from atomsci.ddm.pipeline import sampling as sample

#-----------------------------------------------------------------------------------------------------------------------

class TestApplySamplingMethod(unittest.TestCase):
    def setUp(self):
        # creating a test case 
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([[0], [0], [0], [1], [1]])
        w = np.ones((5, 1))
        ids = np.array([0, 1, 2, 3, 4])
        # create numpy dataset of the values a
        self.train = dc.data.NumpyDataset(X, y, w, ids)

        Params = namedtuple('Params', ['sampling_method', 'sampling_ratio', 'sampling_k_neighbors'])
        self.params_smote = Params(sampling_method='SMOTE', sampling_ratio=1.0, sampling_k_neighbors=1)
        self.params_undersample = Params(sampling_method='undersampling', sampling_ratio=0.9, sampling_k_neighbors=None)
        
        self.random_state = 42
    
    def test_smote(self):
        resampled_data = sample.apply_sampling_method(self.train, self.params_smote, self.random_state)
        
        # check the resampled data
        self.assertEqual(len(resampled_data.X), 6)  # 5 original + 1 synthetic
        self.assertEqual(len(resampled_data.y), 6)
        self.assertEqual(len(resampled_data.w), 6)
        self.assertEqual(len(resampled_data.ids), 6)
        self.assertEqual(sum(resampled_data.y.ravel() == 1), 3)  # 1 original + 2 synthetic
    
    def test_undersampling(self):
        resampled_data = sample.apply_sampling_method(self.train, self.params_undersample, self.random_state)
        
        # Check the resampled data
        self.assertEqual(len(resampled_data.X), 4)  # Reduced to minority class size
        self.assertEqual(len(resampled_data.y), 4)
        self.assertEqual(len(resampled_data.w), 4)
        self.assertEqual(len(resampled_data.ids), 4)
        self.assertEqual(sum(resampled_data.y.ravel() == 1), 2)  # 2 minority classes
    
if __name__ == '__main__':
    unittest.main()