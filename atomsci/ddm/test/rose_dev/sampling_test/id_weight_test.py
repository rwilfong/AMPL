import unittest
import numpy as np
from atomsci.ddm.pipeline import sampling as sample
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
import deepchem as dc

# ------------------------------------------------------------------

class TestApplySamplingMethod(unittest.TestCase):

    def setUp(self):
        # Generate a dummy dataset for testing
        X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, weights=[0.9, 0.1], random_state=42)
        self.train = dc.data.NumpyDataset(X, y, w=np.ones((1000, 1)), ids=np.arange(1000))

        # Define parameters for testing
        class Params:
            def __init__(self, method, ratio=None, neighbors=None):
                self.sampling_method = method
                self.sampling_ratio = ratio
                self.sampling_k_neighbors = neighbors

        self.params_smote = Params('SMOTE', ratio=0.5, neighbors=5)
        self.params_undersampling = Params('undersampling', ratio=0.5)

    def test_train_valid_test_split(self):
        # Perform SMOTE sampling
        random_state = np.random.RandomState(42)
        sampled_train = sample.apply_sampling_method(self.train, self.params_smote)
        
        # Assert correct length of weights and IDs after sampling
        self.assertEqual(len(sampled_train.w), len(sampled_train))
        self.assertEqual(len(sampled_train.ids), len(sampled_train))

        # Perform undersampling
        sampled_train = sample.apply_sampling_method(self.train, self.params_undersampling)
        
        # Assert correct length of weights and IDs after sampling
        self.assertEqual(len(sampled_train.w), len(sampled_train))
        self.assertEqual(len(sampled_train.ids), len(sampled_train))

    def test_k_fold_cv_split(self):
        # Generate K-Fold splits
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for train_index, _ in skf.split(self.train.X, self.train.y):
            train_fold = dc.data.NumpyDataset(self.train.X[train_index], self.train.y[train_index], 
                                              w=self.train.w[train_index], ids=self.train.ids[train_index])

            # Perform SMOTE sampling
            random_state = np.random.RandomState(42)
            sampled_train = sample.apply_sampling_method(train_fold, self.params_smote)
            
            # Assert correct length of weights and IDs after sampling
            self.assertEqual(len(sampled_train.w), len(sampled_train))
            self.assertEqual(len(sampled_train.ids), len(sampled_train))

            # Perform undersampling
            sampled_train = sample.apply_sampling_method(train_fold, self.params_undersampling)
            
            # Assert correct length of weights and IDs after sampling
            self.assertEqual(len(sampled_train.w), len(sampled_train))
            self.assertEqual(len(sampled_train.ids), len(sampled_train))

if __name__ == '__main__':
    unittest.main()
