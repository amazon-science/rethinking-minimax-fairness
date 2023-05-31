# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


#!/usr/bin/env python
# coding: utf-8

"""
Code for ERM and minmax algorithms for different model classes.

Credit
-------

This script uses code for minimax learning from Diana et al. 2021 
and Abernethy et al. 2022 from
the papers' open-source repos on GitHub.

active-sampling-for-minmax-fairness
Project: https://github.com/amazon-science/active-sampling-for-minmax-fairness
Author: Matthäus Kleindessner
License: Apache License, Version 2.0
Code URL https://github.com/amazon-science/active-sampling-for-minmax-fairness/blob/02617eb299fc34de3851b3c88dfcf9d388ca8970/experiment_COMPAS_dataset.py

minimax-fair
Project: https://github.com/amazon-science/minimax-fair
Written by gillwesl-aws and jimmyren23 (Jimmy Ren)
Distributed under Apache 2.0 license
Code URL https://github.com/amazon-science/minimax-fair/blob/a237326f10bb752f6d6c8e5e9e4bb6fc849b3427/main_driver.py
"""

import numpy as np
import torch

from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression, Perceptron
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from src.torch_wrapper import MLPClassifierGPU
from src.mlp_wrapper import TabularMLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest

from src.paired_regression_classifier import PairedRegressionClassifier
from algorithms.algorithm_3 import algorithm_3
from src.minmaxML import do_learning
from metrics import evaluate_model

_REG_PARAM_ALG3 = 0

class ModelWrapperTorchToSklearn:
    def __init__(self, base_model) -> None:
        self.base_model = base_model

    def predict_proba(self, X):
        """
        Assumes base_model expects a torch.Size([n, d]) tensor and returns a torch.Size([n, 2]) tensor
        """
        X = torch.Tensor(X)
        prob = torch.sigmoid(self.base_model(X))
        return prob.cpu().detach().numpy()

    def predict(self, X):
        prob = self.predict_proba(X)
        return (prob[:,1]>0.5).astype(int)

def model_object_from_name(model_type, dim):
    """
    :param dim: number of columns of X
    """
    if model_type == 'LogisticRegression':
        model = LogisticRegression(C=1e5)  # larger C means less regularization
    elif model_type == 'LogisticRegressionSGD':
        model = SGDClassifier(loss='log_loss', penalty=None)
    elif model_type == 'XGBoost':
        model = XGBClassifier()
    elif model_type == 'XGBoostCV':
        model = GridSearchCV(
            estimator = XGBClassifier(random_state = 1, n_jobs = -1),
            param_grid = {'n_estimators': [100,500], 'learning_rate': [0.01,0.1],
            },
            cv = 2,
            refit = True
        )
    elif model_type == 'GradientBoost':
        model = GradientBoostingClassifier(loss='log_loss', max_depth=6)
    elif model_type == 'HistGradientBoost':
        model = HistGradientBoostingClassifier()
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(criterion='log_loss')
    elif model_type == 'RandomForestCV':
        model = GridSearchCV(
            estimator = RandomForestClassifier(random_state = 1, n_jobs = -1),
            param_grid = {'max_features': range(3,dim,3), 'n_estimators': [100,500],
            },
            cv = 2,
            refit = True
        )
    elif model_type == 'DecisionTree2':
        model = DecisionTreeClassifier(criterion='log_loss', max_depth=2)
    elif model_type == 'DecisionTree4':
        model = DecisionTreeClassifier(criterion='log_loss', max_depth=4)
    elif model_type == 'DecisionTree8':
        model = DecisionTreeClassifier(criterion='log_loss', max_depth=8)
    elif model_type == 'LinearSVC':
        model = CalibratedClassifierCV(
                    base_estimator=LinearSVC(),
                    cv=2
                )
    elif model_type == 'KitchenSinkLogistic10':
        model = make_pipeline(RBFSampler(gamma=1, n_components=200, random_state=1), 
                              SelectKBest(dummy_scoring_fn, k=10),
                              SGDClassifier(loss='log_loss', penalty=None))
    elif model_type == 'KitchenSinkLogistic50':
        model = make_pipeline(RBFSampler(gamma=1, n_components=200, random_state=1),
                              SelectKBest(dummy_scoring_fn, k=50),
                              SGDClassifier(loss='log_loss', penalty=None))
    elif model_type == 'KitchenSinkLogistic100':
        model = make_pipeline(RBFSampler(gamma=1, n_components=200, random_state=1),
                              SelectKBest(dummy_scoring_fn, k=100),
                              SGDClassifier(loss='log_loss', penalty=None))
    elif model_type == 'KitchenSinkLogistic200':
        model = make_pipeline(RBFSampler(gamma=1, n_components=200, random_state=1), 
                              SelectKBest(dummy_scoring_fn, k='all'),
                              SGDClassifier(loss='log_loss', penalty=None))
    elif model_type == 'MLPClassifier':
        model = MLPClassifierGPU(h_sizes=[dim,1024,1024], lr=0.001, momentum=0.9, weight_decay=0)  # two hidden layer
    elif model_type == 'TabularMLPClassifier':
        model = TabularMLPClassifier(d_in=dim, d_layers=[1024,1024], d_out=1, lr=0.001, weight_decay=0.0)
    elif model_type == 'PairedRegressionClassifier':
        model = PairedRegressionClassifier(regressor_class=LinearRegression)
    elif model_type == 'LogisticRegressionSGDdim2':
        model = make_pipeline(SelectKBest(dummy_scoring_fn, k=min(dim,2)),
                              SGDClassifier(loss='log_loss', penalty=None))
    elif model_type == 'LogisticRegressionSGDdim4':
        model = make_pipeline(SelectKBest(dummy_scoring_fn, k=min(dim,4)),
                              SGDClassifier(loss='log_loss', penalty=None))
    elif model_type == 'LogisticRegressionSGDdim8':
        model = make_pipeline(SelectKBest(dummy_scoring_fn, k=min(dim,8)),
                              SGDClassifier(loss='log_loss', penalty=None))
    elif model_type == 'LogisticRegressionSGDdim16':
        model = make_pipeline(SelectKBest(dummy_scoring_fn, k=min(dim,16)),
                              SGDClassifier(loss='log_loss', penalty=None))
    elif model_type == 'DummyClassifier':
        model = DummyClassifier(strategy='stratified', random_state=1)
    else:
        raise NotImplementedError(f"model {model_type}")
    return model

# use in kitchen sink and logistic regression to select first n features
def dummy_scoring_fn(X, y):
    return -1*np.arange(X.shape[1])

def train_erm(X, y, model_type):
    dim = X.shape[1]
    model = model_object_from_name(model_type, dim)
    if model_type in ['MLPClassifier']:
        model.fit(X, y, sampleweights=np.ones((X.shape[0],)), n_epochs=10000)
    elif model_type in ['TabularMLPClassifier']:
        model.fit(X, y, sampleweights=np.ones((X.shape[0],)), n_epochs=2000)
    else:
        model.fit(X, y)
    print(f"Completed fitting {model_type} on X of size {X.shape}")
    if model_type in ['XGBoostCV','RandomForestCV']:
        print(f"Best params in GridSearchCV for {model_type}: {model.best_params_}")
    metrics = evaluate_model(model, X, y)
    return model, metrics

def train_minmax(args, X, y, group, group_names, dataset_name, model_type, algo_type, classification_models):
    if algo_type == 'active_sampling_paper':
        assert (model_type==None) or (model_type=="LogisticRegressionSGD"), "active_sampling_paper only implemented for logistic regression"
        feature_transformer = StandardScaler()
        feature_transformer.fit(X)
        X = feature_transformer.transform(X)
        regularization_parameter = _REG_PARAM_ALG3
        nr_steps_Alg3 = args.steps_alg3
        number_of_groups = len(np.unique(group))
        _, _, _, _, _, _, _, model = algorithm_3(
            X, y, group,
            number_of_groups, regularization_parameter=regularization_parameter,
            nr_iterations=nr_steps_Alg3)
        model = make_pipeline(feature_transformer, ModelWrapperTorchToSklearn(model))
        print(f"Completed fitting {algo_type} on X of size {X.shape}")
        metrics = evaluate_model(model, X, y)
    elif algo_type == 'minimax_fair_paper':
        grouplabels = np.expand_dims(group, axis=0)  # required shape (1, n)
        group_names = [group_names]  # required to be a list of lists like [['male','female']]

        a = 1  # Multiplicative coefficient on parametrized learning rate
        b = 1 / 2  # Negative exponent on parameterized learning rate
        scale_eta_by_label_range = True  # Multiplies `a` by square of max abs. label value, to 'normalize' regression labels
        equal_error = False  # Defaults to False for minimax. Set to True to find equal error solution
        error_type = args.error_minimax_fair  # 'MSE', '0/1 Loss', 'FP', 'FN', 'Log-Loss', 'FP-Log-Loss', 'FN-Log-Loss'
        extra_error_types = {}  # Set of additional error types to plot from (only relevant for classification)
        pop_error_type = ''  # Error type for the population on the trajectory (set automatically in general)
        test_size = 0.0  # The proportion of the training data to be withheld as validation data (set to 0.0 for no validation)
        random_split_seed = 4235255  # If test_string1 size > 0.0, the seed to be passed to numpy for train/test split
        
        # Use these arguments to run a single relaxed simulation with on gamma settting
        relaxed = False  # Determines if single run
        gamma = 0.0  # Max groups error if using relaxed variant

        # Solver Specific Settings

        # NOTE Not used. Settings for Logistic Regression (if used)
        logistic_solver = 'liblinear'  # Which logistic regression solver algorithm to use from sklearn (liblinear recommended)
        tol = 1e-15  # The tolerance on the gradient for logistic regression convergence, sklearn default is 1-e4
        max_logi_iters = 100000  # Maximum iterations of logistic regression algorithm
        penalty = 'l2'  # Regularization penalty for log-loss: 'none', 'l1', 'l2'
        C = 1e15  # Inverse of regularization strength, ignored when penalty = 'none'. Set to 1e15 to simulate no regularization

        # NOTE Not used apart from n_epochs. Settings for Multi-Layer Perceptrons (if used)
        # NOTE: Current implementation uses ReLU for all hidden layers and sigmoid for output layer
        if model_type=='MLPClassifier':
            n_epochs = 10000
        else:
            n_epochs = 2000
        lr = None
        momentum = None
        weight_decay = None
        # Hidden sizes is a list denoting the size of each hidden layer in the MLP. Fractional values in the list represent
        # proportions of the input layer, and whole numbers represent absolute layer sizes.
        hidden_sizes = None

        # Plot/output settings
        verbose = True  # enables verbose output for doLearning
        display_plots = False
        show_legend = True  # Denotes if the plots show the legend 
        use_input_commands = False  # Enables 'input(next)' to delay plots until entry into terminal
        dirname = f"auto-MinimaxFair-Results-{args.dir}"  # Specifies which directory to save to, recommend to prefix with 'auto-'
        # NOTE: Use dirname == '' or 'auto-<OUTER DIRECTORY>' to use automatically generated inner folder name

        # Data saving settings
        save_plots = False  # If True, saves plots as PNGs to `dirname` directory (recommended since plots dissapear otherwise)
        save_models = False  # (MEMORY INTENSIVE: not recommended) saves models to `dirname` directory and returns them as list
        return_models = True

        # MODEL/SIMULATION Settings
        fit_intercept = True  # NOTE Not used. If the linear model should fit an intercept (applies only to LinReg and Logreg)
        convergence_threshold = 1e-12  # Converge early if max change in sampleweights between rounds is less than threshold
        group_types = ['Type 1']
        data_name = dataset_name
        warm_start = args.warm_start_minimax_fair
        warm_start_weight = 0.9

        dim = X.shape[1]
        model_class = model_object_from_name(model_type, dim)  # returns an object of model class

        _, _, _, _, _, _, _, _, modelhats, _, _, _, _, _ = \
            do_learning(X, y, args.steps_minimax_fair, grouplabels, a, b, equal_error=equal_error,
                        scale_eta_by_label_range=scale_eta_by_label_range, model_type=model_type,
                        group_names=group_names, group_types=group_types, data_name=data_name,
                        gamma=gamma, relaxed=relaxed, random_split_seed=random_split_seed,
                        verbose=verbose, use_input_commands=use_input_commands,
                        error_type=error_type, extra_error_types=extra_error_types, pop_error_type=pop_error_type,
                        convergence_threshold=convergence_threshold,
                        show_legend=show_legend, save_models=save_models,
                        display_plots=display_plots,
                        test_size=test_size,
                        fit_intercept=fit_intercept, logistic_solver=logistic_solver,
                        max_logi_iters=max_logi_iters, tol=tol, penalty=penalty, C=C,
                        n_epochs=n_epochs, lr=lr, momentum=momentum, weight_decay=weight_decay, hidden_sizes=hidden_sizes,
                        save_plots=save_plots, dirname=dirname,
                        model_class=model_class,
                        classification_models=classification_models, return_models=return_models, 
                        warm_start=warm_start, warm_start_weight=warm_start_weight)
        model = modelhats[-1]  # evaluate model at the last iteration
        print(f"Completed fitting {algo_type} on X of size {X.shape}")
        metrics = evaluate_model(model, X, y)
    else:
        raise NotImplementedError(f'model type {algo_type}')
    return model, metrics