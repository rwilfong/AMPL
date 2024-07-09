# dedicated to the sampling method development playground! 
# ========================================== LIBRARIES ============================================
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

# =========================================== METHODS ==============================================

def apply_sampling_method_dev(train, params, random_state=None, seed=None):
    """Apply a sampling method to a classification dataset
    Inputs: 
        - train: a dc.data.NumpyDataset object
        - params: a parameter object with attributes 'sampling_ratio', 'sampling_k_neighbors', and 'sampling_method'
    Returns:
        - train_resampled: a dc.data.NumpyDataset object with resampled data
    """
    
    sampling_ratio = params.sampling_ratio
    sampling_k_neighbors = params.sampling_k_neighbors
    
    if params.sampling_method == 'SMOTE':
        smote = SMOTE(sampling_strategy=sampling_ratio, k_neighbors=sampling_k_neighbors, random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(train.X, train.y)
        
        # Calculate synthetic weights
        num_original = len(train.X)
        num_synthetic = len(X_resampled) - num_original
        average_weight = np.mean(train.w)
        synthetic_weights = np.full((num_synthetic, 1), average_weight, dtype=np.float64)
        resampled_weights = np.concatenate([train.w, synthetic_weights])
        
        # Update the id length
        synthetic_ids = [f"synthetic_{i}" for i in range(num_synthetic)]
        new_ids = np.concatenate([train.ids, synthetic_ids])
        
    elif params.sampling_method == 'undersampling':
        undersampler = RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=random_state)
        X_resampled, y_resampled = undersampler.fit_resample(train.X, train.y)
        
        # Adjust weights and ids
        resampled_indices = undersampler.sample_indices_
        resampled_weights = train.w[resampled_indices]
        new_ids = train.ids[resampled_indices]
    
    else:
        raise ValueError(f"Unknown sampling method: {params.sampling_method}. Supported methods are 'SMOTE' and 'undersampling'.")
    
    # Create a new dc.data.NumpyDataset with the resampled data
    train_resampled = dc.data.NumpyDataset(X_resampled, y_resampled, train.w, train.ids)
    
    
    print("Resampled! Moving on...")
    return train_resampled

# ======================================== K FOLD DEV =============================================

def apply_sampling_method_k_fold_cv(train, params):
    sampling_ratio = params.sampling_ratio
    
    #for train, valid in train_valid_dsets:
    if params.sampling_method == 'SMOTE':
        sampling_k_neighbors = params.sampling_k_neighbors
        smote = SMOTE(sampling_strategy=sampling_ratio, k_neighbors=sampling_k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(train.X, train.y)
        print(train)
            
        # Calculate synthetic weights
        num_original = len(train.X)
        num_synthetic = len(X_resampled) - num_original
        average_weight = np.mean(train.w)
        synthetic_weights = np.full((num_synthetic, 1), average_weight, dtype=np.float64)
        resampled_weights = np.concatenate([train.w, synthetic_weights])
            
        # Update the id length 
        synthetic_ids = [f"synthetic_{i}" for i in range(num_synthetic)]
        new_ids = np.concatenate([train.ids, synthetic_ids])
        
    elif params.sampling_method == 'undersampling':
        undersampler = RandomUnderSampler(sampling_strategy=sampling_ratio)
        X_resampled, y_resampled = undersampler.fit_resample(train.X, train.y)
            
        # Adjust weights and ids 
        resampled_indices = undersampler.sample_indices_
        resampled_weights = train.w[resampled_indices]
        new_ids = train.ids[resampled_indices]

    else:
        raise ValueError(f"Unknown sampling method: {params.sampling_method}. Supported methods are 'SMOTE' and 'undersampling'.")
        
    # Create a new dc.data.NumpyDataset with the resampled data
    resampled_train = dc.data.NumpyDataset(X_resampled, y_resampled, resampled_weights, new_ids)
        
    # Append the resampled train and original valid datasets as a tuple
    #resampled_train_valid_dsets.append((resampled_train, valid))
    
    return resampled_train

# =====================================================================================================

def apply_sampling_method_k_fold_cv(train_folds, params):
    sampling_ratio = params.sampling_ratio
    sampling_k_neighbors = params.sampling_k_neighbors
    
    if params.sampling_method == 'SMOTE':
        smote = SMOTE(sampling_strategy=sampling_ratio, k_neighbors=sampling_k_neighbors)
    elif params.sampling_method == 'undersampling':
        undersampler = RandomUnderSampler(sampling_strategy=sampling_ratio)
    else:
        raise ValueError(f"Unknown sampling method: {params.sampling_method}. Supported methods are 'SMOTE' and 'undersampling'.")
    
    folds_X_resampled = []
    folds_y_resampled = []
    folds_weights = []
    folds_ids = []
    
    for fold_X, fold_y, fold_w, fold_ids in train_folds:
        if params.sampling_method == 'SMOTE':
            X_resampled, y_resampled = smote.fit_resample(fold_X, fold_y)
            num_original = len(fold_X)
            num_synthetic = len(X_resampled) - num_original
            average_weight = np.mean(fold_w)
            synthetic_weights = np.full((num_synthetic, 1), average_weight, dtype=np.float64)
            resampled_weights = np.concatenate([fold_w, synthetic_weights])
            synthetic_ids = [f"synthetic_{i}" for i in range(num_synthetic)]
            new_ids = np.concatenate([fold_ids, synthetic_ids])
            
        elif params.sampling_method == 'undersampling':
            X_resampled, y_resampled = undersampler.fit_resample(fold_X, fold_y)
            resampled_indices = undersampler.sample_indices_
            resampled_weights = fold_w[resampled_indices]
            new_ids = fold_ids[resampled_indices]
        
        folds_X_resampled.append(X_resampled)
        folds_y_resampled.append(y_resampled)
        folds_weights.append(resampled_weights)
        folds_ids.append(new_ids)
    
    return folds_X_resampled, folds_y_resampled, folds_weights, folds_ids