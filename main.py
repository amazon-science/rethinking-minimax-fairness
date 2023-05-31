# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


#!/usr/bin/env python
# coding: utf-8

"""
Fit classification models using ERM and minmax learning method.
Evaluate models on their worst-off group performance on multiple datasets.

Datasets: Uses US Census Income data curated by folktables package, 
UCI Adult Income, Compas, Drug consumption, Diabetes, Default, Communities, 
Credit, Heart, and Marketing.

Run commands:
`python main.py --dir 'alldata_allmodel' --dataset_id -1 --model_id -1 --steps_minimax_fair 10000 --to_categorical --warm_start_minimax_fair`
Change `dataset_id` and `model_id` to the specific id of the dataset and model
Both `dataset_id` and `model_id` are 1-indexed
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve

from metrics import get_fraction_similar_error
from metrics import evaluate_model
from models import train_erm, train_minmax

from dataloader import Dataset

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
sns.set_palette('colorblind')

STORAGE_PATH = './results'  # saves logs, plots in this directory

# List of base models to use. Specific ones to use are passed as an argument
# NOTE: keep MLP as the first classifier as it is assumed by the command-line arguments
MODEL_TYPES = [
    'TabularMLPClassifier',
    'LogisticRegressionSGD',
    'LogisticRegressionSGDdim2',
    'LogisticRegressionSGDdim4',
    'LogisticRegressionSGDdim8',
    'DecisionTree2',
    'DecisionTree4',
    'DecisionTree8',
    'RandomForest',
    'LinearSVC',
    # 'GradientBoost',
    ]  # Options 'MLPClassifier','XGBoost','LogisticRegressionSGDdim16','LogisticRegression','HistGradientBoost','Perceptron','KitchenSinkLogistic10','DummyClassifier','PairedRegressionClassifier', 'KitchenSinkLogistic50','KitchenSinkLogistic100','KitchenSinkLogistic200',

##########################
######## HELPERS
##########################

def create_outdir(args):
    """Creates directory to save results."""
    now = datetime.now()
    dt = now.strftime("%Y-%m-%d %Hh%Mm%Ss%fms")
    args.dir = os.path.join(os.path.join(STORAGE_PATH, args.dir), dt)
    Path(args.dir).mkdir(parents=True, exist_ok=True)
    torch.save(args, os.path.join(args.dir, 'args.pt'))
    with open(os.path.join(args.dir, 'cmd.txt'), 'w') as fw:
        json.dump(args.__dict__, fw)
    return args

def format_result(group_labels_train_on, group_labels_evaluate_on, evals):
    """Flattens the results array into one entry for each metric."""
    d = []
    for g_train_on, g_eval_on, eval in zip(group_labels_train_on, group_labels_evaluate_on, evals):
        d += [[('metric', metric), ('trained_on', g_train_on), ('evaluated_on', f'group_{g_eval_on}'), ('value', value)] for (metric, value) in eval.items()]
    return d

def print_data_statistics(args, dataset_names):
    """Compute stats such as number of rows, features, groups in datasets.""" 
    stats = []
    for dataset_name, group_type in dataset_names:
        print(f'\n\n=========Starting dataset {dataset_name}========\n\n')
        features, label, group, features_group_removed, _, _ = data_handler.read_dataset(args, dataset_name, group_type)
        X_train, X_test, _, _, group_train, group_test, X_group_removed_train, X_group_removed_test = train_test_split(
            features, label, group, features_group_removed, test_size=0.3, random_state=1)
        stats.append({'dataset_name': dataset_name, 'group_type': group_type,
            'train_rows': X_train.shape[0], 'test_rows': X_test.shape[0],
            'total_rows': X_train.shape[0]+X_test.shape[0],
            'train_columns': X_train.shape[1], 'test_columns': X_test.shape[1],
            'train_minus_group_columns': X_group_removed_train.shape[1], 'test_minus_group_columns': X_group_removed_test.shape[1],
            'train_groups': len(np.unique(group_train)), 'test_groups': len(np.unique(group_test)),
            'train_group_proportion': pd.DataFrame(group_train).value_counts(normalize=True), 'test_group_proportion': pd.DataFrame(group_test).value_counts(normalize=True),
        })
        print(f"\n\nSamples train {X_train.shape}, test {X_test.shape}")
        print(f"\n===Data points per group train===\n{pd.DataFrame(group_train).value_counts()},\ntest\n{pd.DataFrame(group_test).value_counts()}")
    return pd.DataFrame(stats)

##################################
######## Training and evaluation
##################################

def train_group_models(X, y, group, X_group_removed, full_model, minmax_model, model_type):
    """
    Train group-specific models for each group by ERM. 
    Evaluate group model, full ERM, and minmax model on the same group.

    Args:
        X, y: features, class labels
        group: group labels
        X_group_removed: features without the group label
        full_model: ERM on all data
        minmax_model: minmax model
        model_type: model class for learners
    """
    group_models = []
    group_evals = []
    full_evals = []
    minmax_evals = []
    group_labels_train_on = []
    group_labels_evaluate_on = []

    # Train and evaluate on same group
    for g in list(np.unique(group)):
        X_subset = X[group==g]
        y_subset = y[group==g]
        X_group_removed_subset = X_group_removed[group==g]
        group_model, group_eval = train_erm(X_subset, y_subset, model_type)  # group-specific model i.e. ERM on a group's data
        full_eval = evaluate_model(full_model, X_subset, y_subset)
        minmax_eval = evaluate_model(minmax_model, X_group_removed_subset, y_subset)

        # Metric which computes the fraction of data points with similar error
        similar_error_minmax = get_fraction_similar_error(X_subset, y_subset, X_group_removed_subset, full_model, minmax_model)
        full_eval['fraction_similar_error'] = similar_error_minmax
        minmax_eval['fraction_similar_error'] = similar_error_minmax
        similar_error_group = get_fraction_similar_error(X_subset, y_subset, X_subset, full_model, group_model)
        group_eval['fraction_similar_error'] = similar_error_group

        group_models.append(group_model)
        group_evals.append(group_eval)
        full_evals.append(full_eval)
        minmax_evals.append(minmax_eval)
        group_labels_train_on.append(g)
        group_labels_evaluate_on.append(g)
    
    # Compute group calibration
    ece_each_group_full = [full_eval['calibration'] for full_eval in full_evals]  # assumes train and evaluate on same groups
    calibration_group_mean = np.mean(ece_each_group_full)
    calibration_group_std = np.std(ece_each_group_full)
    calibration_group_worst = max(ece_each_group_full, key=lambda x: abs(x-1))
    for id in range(len(full_evals)):
        full_evals[id]['calibration_group_mean'] = calibration_group_mean
        full_evals[id]['calibration_group_std'] = calibration_group_std
        full_evals[id]['calibration_group_worst'] = calibration_group_worst
    ece_each_group_minmax = [minmax_eval['calibration'] for minmax_eval in minmax_evals]
    calibration_group_mean = np.mean(ece_each_group_minmax)
    calibration_group_std = np.std(ece_each_group_minmax)
    calibration_group_worst = max(ece_each_group_minmax, key=lambda x: abs(x-1))
    for id in range(len(minmax_evals)):
        minmax_evals[id]['calibration_group_mean'] = calibration_group_mean
        minmax_evals[id]['calibration_group_std'] = calibration_group_std
        minmax_evals[id]['calibration_group_worst'] = calibration_group_worst

    evals = []
    evals += format_result([f'group_{g}' for g in group_labels_train_on], 
                            group_labels_evaluate_on, group_evals)
    evals += format_result([f'full' for _ in group_labels_evaluate_on], 
                            group_labels_evaluate_on, full_evals)
    evals += format_result([f'minmax' for _ in group_labels_evaluate_on], 
                            group_labels_evaluate_on, minmax_evals)

    return group_labels_train_on, group_models, evals

def evaluate_group_models(X, y, group, X_group_removed, group_labels_train_on, models, full_model, minmax_model):
    """
    Evaluate pretrained models on each group.
    
    Args:
        X, y: features, class labels
        X_group_removed: features without the group label
        group_labels_train_on: group indices in the same order 
            as group-spefic models in `models`
        group: group labels
        models: group-specific models
        full_model: ERM on all data
        minmax_model: minmax model
    """
    group_evals = []
    full_evals = []
    minmax_evals = []
    for g, model in zip(group_labels_train_on, models):
        X_subset = X[group==g]
        y_subset = y[group==g]
        X_group_removed_subset = X_group_removed[group==g]
        group_eval = evaluate_model(model, X_subset, y_subset)
        full_eval = evaluate_model(full_model, X_subset, y_subset)
        minmax_eval = evaluate_model(minmax_model, X_group_removed_subset, y_subset)

        # Metric which computes the fraction of data points with similar error
        similar_error_minmax = get_fraction_similar_error(X_subset, y_subset, X_group_removed_subset, full_model, minmax_model)
        full_eval['fraction_similar_error'] = similar_error_minmax
        minmax_eval['fraction_similar_error'] = similar_error_minmax
        similar_error_group = get_fraction_similar_error(X_subset, y_subset, X_subset, full_model, model)
        group_eval['fraction_similar_error'] = similar_error_group

        group_evals.append(group_eval)
        full_evals.append(full_eval)
        minmax_evals.append(minmax_eval)
    
    # Compute group calibration
    ece_each_group_full = [full_eval['calibration'] for full_eval in full_evals]  # assumes train and evaluate on same groups
    calibration_group_mean = np.mean(ece_each_group_full)
    calibration_group_std = np.std(ece_each_group_full)
    calibration_group_worst = max(ece_each_group_full, key=lambda x: abs(x-1))
    for id in range(len(full_evals)):
        full_evals[id]['calibration_group_mean'] = calibration_group_mean
        full_evals[id]['calibration_group_std'] = calibration_group_std
        full_evals[id]['calibration_group_worst'] = calibration_group_worst
    ece_each_group_minmax = [minmax_eval['calibration'] for minmax_eval in minmax_evals]
    calibration_group_mean = np.mean(ece_each_group_minmax)
    calibration_group_std = np.std(ece_each_group_minmax)
    calibration_group_worst = max(ece_each_group_minmax, key=lambda x: abs(x-1))
    for id in range(len(minmax_evals)):
        minmax_evals[id]['calibration_group_mean'] = calibration_group_mean
        minmax_evals[id]['calibration_group_std'] = calibration_group_std
        minmax_evals[id]['calibration_group_worst'] = calibration_group_worst

    evals = []
    evals += format_result([f'group_{g}' for g in group_labels_train_on], 
                            group_labels_train_on, group_evals)
    evals += format_result([f'full' for _ in group_labels_train_on], 
                            group_labels_train_on, full_evals)
    evals += format_result([f'minmax' for _ in group_labels_train_on], 
                            group_labels_train_on, minmax_evals)

    return group_labels_train_on, evals

def evaluate_all_pairs_group_models(X, y, group, X_group_removed, group_labels_train_on_unique, group_models, full_model, minmax_model):
    """Evaluate pretrained models for each group on every group."""
    group_evals = []
    full_evals = []
    minmax_evals = []
    group_labels_train_on = []
    group_labels_evaluate_on = []
    for g_evaluate_on in group_labels_train_on_unique:
        X_subset = X[group==g_evaluate_on]
        y_subset = y[group==g_evaluate_on]
        X_group_removed_subset = X_group_removed[group==g_evaluate_on]
        # Full and Minmax models
        full_eval = evaluate_model(full_model, X_subset, y_subset)
        minmax_eval = evaluate_model(minmax_model, X_group_removed_subset, y_subset)
        full_evals.append(full_eval)
        minmax_evals.append(minmax_eval)
        # Group models
        for g_train_on, group_model in zip(group_labels_train_on_unique, group_models):
            group_eval = evaluate_model(group_model, X_subset, y_subset)
            group_evals.append(group_eval)
            group_labels_train_on.append(g_train_on)
            group_labels_evaluate_on.append(g_evaluate_on)
    
    evals = []
    evals += format_result([f'group_{g}' for g in group_labels_train_on], 
                            group_labels_evaluate_on, group_evals)
    evals += format_result([f'full' for _ in group_labels_train_on_unique], 
                            group_labels_train_on_unique, full_evals)
    evals += format_result([f'minmax' for _ in group_labels_train_on_unique], 
                            group_labels_train_on_unique, minmax_evals)
    return evals

def run_model_type(X_train, X_test, y_train, y_test, group_train, group_test, X_group_removed_train, X_group_removed_test, minmax_model, model_type, dataset_name, group_type):
    results = []
    # Train full data model - ERM
    full_model, _ = train_erm(X_train, y_train, model_type)
    
    # Train group-specific models, evaluate on TRAIN data
    group_labels_train_on, group_models, evals = train_group_models(X_train, y_train, group_train, X_group_removed_train, full_model, minmax_model, model_type)
    params = [
        ('dataset', dataset_name), ('model_type', model_type),
         ('group_type', group_type), ('eval_data_type', 'train')
    ]
    result = [dict(params + i) for i in evals]
    results += result

    # Evaluate group-specific models on TEST data
    _, evals_test = evaluate_group_models(X_test, y_test, group_test, X_group_removed_test, group_labels_train_on, group_models, full_model, minmax_model)
    params = [
        ('dataset', dataset_name), ('model_type', model_type), 
        ('group_type', group_type), ('eval_data_type', 'test')
    ]
    result = [dict(params + i) for i in evals_test]
    results += result

    evals_all_pairs = evaluate_all_pairs_group_models(X_train, y_train, group_train, X_group_removed_train, group_labels_train_on, group_models, full_model, minmax_model)
    params = [
        ('dataset', dataset_name), ('model_type', model_type), 
        ('group_type', group_type), ('eval_data_type', 'train')
    ]
    results_all_pairs = [dict(params + i) for i in evals_all_pairs]

    return results, results_all_pairs

def eval_group_predictor(dataset_name, group_type, X_train, X_test, y_train, y_test, group_train, group_test, group_label_map):
    '''
    Train and evaluate classifiers to predict group label.
    Uses Random Forest classifier.
    '''
    results = []

    group_predictor = RandomForestClassifier().fit(X_train, group_train)
    group_predictor_train_eval = evaluate_model(group_predictor, X_train, group_train)
    params = [
        ('dataset', dataset_name), ('model_type', 'randomforest'), 
        ('group_type', group_type), ('eval_data_type', 'train')
    ]
    d = [[('metric', metric), ('value', value)] for (metric, value) in group_predictor_train_eval.items()]
    result = [dict(params + i) for i in d]
    results += result

    # Precision-recall curve Train
    for g_id, g_name in enumerate(np.unique(group_train)):
        precision, recall, _ = precision_recall_curve(
            group_train==g_name, group_predictor.predict_proba(X_train)[:, g_id]
        )
        plt.plot(recall, precision, lw=2, label=f"Group {g_name}")
    plt.legend(loc="best")
    plt.title(f"Precision-Recall curve Train {dataset_name}\nMap={group_label_map}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(os.path.join(args.dir, f'precision_recall_curve_{dataset_name}_{group_type}_group_predictor_randomforest_train.png'), bbox_inches='tight')
    plt.close()

    group_predictor_test_eval = evaluate_model(group_predictor, X_test, group_test)
    params = [
        ('dataset', dataset_name), ('model_type', 'randomforest'), 
        ('group_type', group_type), ('eval_data_type', 'test')
    ]
    d = [[('metric', metric), ('value', value)] for (metric, value) in group_predictor_test_eval.items()]
    result = [dict(params + i) for i in d]
    results += result

    # Precision-recall curve Test
    for g_id, g_name in enumerate(np.unique(group_train)):
        precision, recall, _ = precision_recall_curve(
            group_test==g_name, group_predictor.predict_proba(X_test)[:, g_id]
        )
        plt.plot(recall, precision, lw=2, label=f"Group {g_name}")
    plt.legend(loc="best")
    plt.title(f"Precision-Recall curve Test {dataset_name}\nMap={group_label_map}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(os.path.join(args.dir, f'precision_recall_curve_{dataset_name}_{group_type}_group_predictor_randomforest_test.png'), bbox_inches='tight')
    plt.close()
    return results

##########################
######## Main functions
##########################

def run_dataset(args, data_handler, dataset_name_and_group, model_types):
    """
    Train and evaluate full data ERM, group-specific ERM, and minmax models
    for all model types on a single dataset.

    Args:
        args: run arguments
        data_handler: object to query dataset info
        dataset_name_and_group: tuple of dataset name and
            which group (like gender, race) to use
        model_types: list of base models
    """
    # Prepare data
    dataset_name, group_type = dataset_name_and_group
    features, label, group, features_group_removed, group_names, group_label_map = data_handler.read_dataset(args, dataset_name, group_type)
    X_train, X_test, y_train, y_test, group_train, group_test, X_group_removed_train, X_group_removed_test = train_test_split(
        features, label, group, features_group_removed, test_size=0.3, random_state=1)
    print(f"Samples train {X_train.shape}, test {X_test.shape}")
    print(f"\n===Data points per group train===\n{pd.DataFrame(group_train).value_counts()},\ntest\n{pd.DataFrame(group_test).value_counts()}")
    
    # Scale each feature by subtracting mean and dividing standard deviation
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    scaler.fit(X_group_removed_train)
    X_group_removed_train = scaler.transform(X_group_removed_train)
    X_group_removed_test = scaler.transform(X_group_removed_test)

    args.group_label_map = group_label_map  # to save group mapping in args file

    if args.exclude_group_feature:
        X_train, X_test = X_group_removed_train, X_group_removed_test
    
    # Ensure all models fitted and evaluated on the same data
    results = []
    results_all_pairs = []

    for model_type in model_types:
        print(f'\n===Running model {model_type}===\n')
        # Minmax training
        start_time = time.time()
        # Abernethy et al. minmax method
        # Only works for logistic regression model
        # minmax_model, _ = train_minmax(args, X_group_removed_train, y_train, group_train, group_names, 
        #                         dataset_name, model_type=None, algo_type='active_sampling_paper', classification_models=MODEL_TYPES)  # train minmax model once and use in all model_type evaluations
        
        # Diana et al. minmax method
        minmax_model, _ = train_minmax(args, X_train, y_train, group_train, group_names, 
                                dataset_name, model_type, algo_type='minimax_fair_paper', classification_models=MODEL_TYPES)  # keep group feature in X_train for minmax models
        end_time = time.time()
        print(f'Time taken by minmax is {end_time-start_time} seconds')
        # ERM training on full and per-group data
        result, result_all_pairs = run_model_type(X_train, X_test, y_train, y_test, group_train, group_test, 
                                        X_train, X_test, minmax_model, model_type, dataset_name, group_type)  # keep group feature in X_train, X_test for group models
        results += result
        results_all_pairs += result_all_pairs

    # Predicting group information from features
    results_group_predictor = eval_group_predictor(dataset_name, group_type, X_train, X_test, y_train, y_test, group_train, group_test, group_label_map)
    return results, results_group_predictor, results_all_pairs

def main(args, data_handler, dataset_names, model_types):
    """
    Main function to run for all datasets and model types

    Args:
        args: run arguments
        data_handler: object to query dataset info
        dataset_names: list of datasets to run
        model_types: list of models to run
    """
    results = []
    results_group_predictor = []
    results_all_pairs = []
    for dataset_name in dataset_names:
        print(f'\n\n=========Starting dataset {dataset_name}========\n\n')
        result, result_group_predictor, result_all_pairs = run_dataset(args, data_handler, dataset_name, model_types)
        results += result
        results_group_predictor += result_group_predictor
        results_all_pairs += result_all_pairs

        df_single_data = pd.DataFrame(result)
        df_single_data.to_csv(os.path.join(args.dir, f'metrics_did{args.dataset_id}_{dataset_name[0]}_{dataset_name[1]}.csv'), index=False)
        df_group_single_data = pd.DataFrame(result_group_predictor)
        df_group_single_data.to_csv(os.path.join(args.dir, f'metrics_group_predictor_did{args.dataset_id}_{dataset_name[0]}_{dataset_name[1]}.csv'), index=False)
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(args.dir, f'metrics_did{args.dataset_id}_dataset.csv'), index=False)
        df_all_pairs = pd.DataFrame(results_all_pairs)
        df_all_pairs.to_csv(os.path.join(args.dir, f'metrics_all_pairs_did{args.dataset_id}.csv'), index=False)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.dir, f'metrics_did{args.dataset_id}_dataset.csv'), index=False)
    df_group_predictor = pd.DataFrame(results_group_predictor)
    df_group_predictor.to_csv(os.path.join(args.dir, f'metrics_group_predictor_did{args.dataset_id}.csv'), index=False)
    df_all_pairs = pd.DataFrame(results_all_pairs)
    df_all_pairs.to_csv(os.path.join(args.dir, f'metrics_all_pairs_did{args.dataset_id}.csv'), index=False)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='alldata')  # output directory
    parser.add_argument('--dataset_id', default=-1, type=int)  # 1 indexed
    parser.add_argument('--model_id', default=-1, type=int)  # 1 indexed
    parser.add_argument('--data_max_size', default=1000000000, type=int)  # max number of data points in each dataset
    parser.add_argument('--exclude_group_feature', action='store_true')  # do not give group indicator as a feature
    parser.add_argument('--to_categorical', action='store_true')  # use categorical features
    parser.add_argument('--steps_alg3', default=10000, type=int)  # param for minimax algorithm of Abernethy et al. 2022
    parser.add_argument('--steps_minimax_fair', default=10000, type=int)  # param for minimax algorithm of Diana et al. 2021
    parser.add_argument('--error_minimax_fair', default='Log-Loss', type=str)  # Options 'MSE', '0/1 Loss', 'FP', 'FN', 'Log-Loss', 'FP-Log-Loss', 'FN-Log-Loss'
    parser.add_argument('--warm_start_minimax_fair', action='store_true')  # start from a high weight on worst-off group

    args = parser.parse_args()
    print(args)

    data_handler = Dataset()
    if args.dataset_id==-1:  # run for all datasets
        dataset_names = data_handler.list_datasets()
    else:
        dataset_names = data_handler.list_datasets()[args.dataset_id-1:args.dataset_id]  # 1 indexed
    if args.model_id==-1:
        model_types = MODEL_TYPES[1:]  # all models except MLPs
    else:
        model_types = MODEL_TYPES[args.model_id-1:args.model_id]  # 1 indexed
    args.datasets = dataset_names
    args.models = model_types
    args = create_outdir(args)
    
    df = main(args, data_handler, dataset_names, model_types)
    
    stats = print_data_statistics(args, dataset_names)
    stats.to_csv(os.path.join(args.dir, 'dataset_statistics.csv'))
    print(stats)
    
    torch.save(args, os.path.join(args.dir, 'args.pt'))

    sys.exit(0)