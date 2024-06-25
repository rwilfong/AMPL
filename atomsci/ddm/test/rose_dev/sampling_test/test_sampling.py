#!/usr/bin/env python 
"""Testing the sampling methods. Want to ensure that the model pipeline works and that the sampling methods are incorporated.
Based off of the test_kfold_split.py method. """
import pandas as pd 
import numpy as np
import sklearn.metrics as skmetrics 
import copy
import os
import json 

from atomsci.ddm.pipeline import model_pipeline as mp
from atomsci.ddm.pipeline import parameter_parser as parse
from atomsci.ddm.test.integrative import integrative_utilities
import atomsci.ddm.pipeline.predict_from_model as pfm

########### MODEL DEETS ################

def get_test_set(dataset_key, split_csv, id_col):
    """
    Read the dataset key and split_uuid to split dataset into split components 
    
    Parameters: 
        - dataset_key: path to csv file of dataset
        - split_uuid: path to split csv file 
        - id_col: name of ID column
    
    Returns:
        - train, valid, test dataframe
    """
    df = pd.read_csv(dataset_key)
    split_df=pd.read_csv(split_csv)
    test_df = df[df[id_col].isin(split_df[split_df['subset']=='test']['cmpd_id'])]

    return test_df

def split(pparams):
    split_params=copy.copy(pparams)
    split_params.split_only=True
    split_params.previously_split=False

    model_pipeline= mp.ModelPipeline(split_params)
    split_uuid = model_pipeline.split_dataset()

    return split_uuid

def train(pparams):
    train_pipe = mp.ModelPipeline(pparams)
    train_pipe.train_model()

    return train_pipe

def find_best_test_metric(model_metrics):
    for metric in model_metrics:
        if metric['label'] == 'best' and metric['subset']=='test':
            return metric 
    return None 

##########################################################

################# ACTUAL TESTS ###########################

def train_valid_test_RF_random_test(): 
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'train_valid_test_rf_random.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key= os.path.join(script_path,
                                      '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path
    pparams.split_uuid= 'test-split'

    saved_model_identity(pparams)

def train_valid_test_NN_scaffold_test():
    script_path = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_path, 'train_valid_test_NN_scaffold.json')

    pparams = parse.wrapper(['--config_file', json_file])
    pparams.dataset_key= os.path.join(script_path,
                                      '../../test_datasets/aurka_chembl_base_smiles_union.csv')
    pparams.result_dir=script_path
    pparams.split_uuid= 'test-split'

    saved_model_identity(pparams)

###########################################################

################ RETURN INFO ##############################

def saved_model_identity(pparams):
    script_path = os.path.dirname(os.path.realpath(__file__))
    # script path return: /Users/rwilfong/Downloads/2024_LLNL/fork_ampl/AMPL/atomsci/ddm/test/rose_dev/sampling_test
    if not pparams.previously_split:
        split_uuid = split(pparams)
        
        pparams.split_uuid = split_uuid
        pparams.previously_split = True 
    
    train_pipe = train(pparams)
    split_csv = os.path.join(script_path, '../../test_datasets/', train_pipe.data._get_split_key())
    test_df = get_test_set(pparams.dataset_key, split_csv, pparams.id_col)

    # load model metrics from file 
    with open(os.path.join(pparams.output_dir, 'model_metrics.json'), 'r') as f:
        model_metrics = json.load(f)
    
    metrics = find_best_test_metric(model_metrics)
    id_col = metrics['input_dataset']['id_col']
    response_col=metrics['input_dataset']['response_cols'][0]
    smiles_col = metrics['input_dataset']['smiles_col']
    test_length = metrics['prediction_results']['num_compounds']

    model_tar = train_pipe.params.model_tarball_path
    pred_df = pfm.predict_from_model_file(model_tar, test_df, id_col=id_col,
                smiles_col=smiles_col, response_col=response_col)
    pred_df2 = pfm.predict_from_model_file(model_tar, test_df, id_col=id_col,
                smiles_col=smiles_col, response_col=response_col)
    
    X = pred_df[response_col+'_actual'].values
    y = pred_df[response_col+'_pred'].values
    X2 = pred_df2[response_col+'_actual'].values
    y2 = pred_df2[response_col+'_pred'].values
    pred_df.to_csv('/Users/rwilfong/Downloads/2024_LLNL/fork_ampl/AMPL/atomsci/ddm/test/rose_dev/sampling_test/pred_df.csv')
    print('returned pred_df to a csv')
    pred_df2.to_csv('/Users/rwilfong/Downloads/2024_LLNL/fork_ampl/AMPL/atomsci/ddm/test/rose_dev/sampling_test/pred_df2.csv')
    print('returned pred_df2 to a csv')

##########################################################

###### DEV ZONE ##########
def k_fold_cv_NN_test():
    pass

def k_fold_cv_RF_test():
    pass

##########################

if __name__=='__main__':
    script_path = os.path.dirname(os.path.realpath(__file__))
    print(script_path)
    print('train_valid_test_rf_random_test')
    train_valid_test_RF_random_test()
    #print('train_valid_test_NN_scaffold_test')
