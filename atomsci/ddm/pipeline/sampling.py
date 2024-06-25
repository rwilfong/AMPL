"""Functions to perform sampling on classification datasets."""

import pandas as pd 
import numpy as np
import os
# sampling specific libraries 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
# AMPL dependencies 
from atomsci.ddm.pipeline import parameter_parser as pp
from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.pipeline import model_datasets as md 
# deepchem for dataset
import deepchem as dc
# =====================================================================================================
def apply_sampling_method(train, params):
    """Apply a sampling method to a classification dataset when split_strategy=='train_valid_test'
    Inputs: 
        - train: DeepChem NumpyDataset with train.X, train.y, train.w, and train.ids
        - params (NameSpace object): contains all the parameter information.
    Returns:
        - train_resampled: a DeepChem NumpyDataset with train.X, train.y, train.w, and train.ids
    """
    print(f"split type: {params.split_strategy}, sampling method: {params.sampling_method}")
    sampling_ratio = params.sampling_ratio
    if params.sampling_method=='SMOTE':
        # moving k-neighbors into SMOTE since it is SMOTE specific
        sampling_k_neighbors = params.sampling_k_neighbors
        smote=SMOTE(sampling_strategy=sampling_ratio, k_neighbors=sampling_k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(train.X, train.y.ravel())
        y_resampled=y_resampled.reshape(-1, 1)
        
        # calculate synthetic weights.  
        num_original = len(train.X)
        num_synthetic = len(X_resampled)-num_original
        # how much effect do we want resampling to have? Could make a user-specified entry
        # creating an average of the weights and using that to fill in 
        # could also implement weights close to 0 for little to no effect. 
        average_weight = np.mean(train.w)
        synthetic_weights=np.full((num_synthetic,1), average_weight, dtype=np.float64)
        resampled_weights=np.concatenate([train.w, synthetic_weights])
        
        # update the id length 
        synthetic_ids = [f"synthetic_{i}" for i in range(num_synthetic)]
        new_ids = np.concatenate([train.ids, synthetic_ids])
        
    elif params.sampling_method == 'undersampling':
        undersampler = RandomUnderSampler(sampling_strategy=sampling_ratio)
        X_resampled, y_resampled = undersampler.fit_resample(train.X, train.y.ravel())
        y_resampled=y_resampled.reshape(-1, 1)
        
        #adjust weights and ids 
        resampled_indices = undersampler.sample_indices_
        resampled_weights = train.w[resampled_indices]
        new_ids = train.ids[resampled_indices]

    else:
        raise ValueError(f"Unknown sampling method: {params.sampling_method}. Supported methods are 'SMOTE' and 'undersampling'.")
    # return a new dc.data.NumpyDataset with the resampled data, the original weights and ids
    train_resampled= dc.data.NumpyDataset(X_resampled, y_resampled, resampled_weights, new_ids) 
    return train_resampled


