# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


#!/usr/bin/env python
# coding: utf-8

"""
Code to compute metrics for classification

Metrics
    logloss, accuracy, balanced accuracy, Matthews correlation coefficient
    expected calibration error, calibration slope, root Brier score

Expected calibration error metric is computed using netcal package
Rest of metrics are computed using sklearn package
"""

import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score, matthews_corrcoef
from netcal.metrics import ECE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

_ECE_NUMBER_BINS = 15  # arbitrary number of bins, increased from default value of 10

#########################
### Evaluation metrics
#########################

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    if hasattr(model, 'predict_proba'):
        y_pred_prob = model.predict_proba(X)
    else:
        print('predict_proba not defined for', model)
        y_pred_prob = y_pred
    logloss = log_loss(y, y_pred_prob)
    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)
    if len(np.unique(y))!=2:
        ece = None  # expected calibration error only implemented for binary classification
        cslope = None  # calibration slope
        rbs = None  # root Brier score
    else:
        if len(y_pred_prob.shape)>1:
            y_pred_prob = y_pred_prob[:,1]
        ece = ECE(_ECE_NUMBER_BINS, detection=False).measure(y_pred_prob, y)  # requires probability of positive prediction
        y_pred_prob = np.clip(y_pred_prob, a_min=1e-12, a_max=1-1e-12)
        cslope = calibration_slope(y, y_pred_prob)
        rbs = np.sqrt(brier_score_loss(y, y_pred_prob))  # upper bound of true calibration error, ref. https://arxiv.org/abs/2203.07835
    metrics = {
        'neglogloss': -1*logloss,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'matthews_corrcoef': mcc,
        'expected_calibration_error': ece,
        'root_brier_score': rbs,
        'calibration': cslope,  # calibration slope used by default to quantify within-group calibration
        'calibration_group_mean': cslope,  # to be replaced in train_group_models() for erm and minmax models
        'calibration_group_std': None,  # same, dummy value for group-specific models since only one value of ece is possible
        'calibration_group_worst': cslope  # same, dummy value for group-specific models since only one value of ece is possible
    }
    return metrics

def logit(p, eps=1e-6):
    p = np.clip(p, a_min=eps, a_max=1-eps)  # avoids log(0) or log(inf)
    return np.log(p/(1-p))

def calibration_slope(ground_truth, probabilities):
    probabilities = np.array(probabilities)
    logit_probabilities = logit(probabilities).reshape(-1,1)
    lr = LogisticRegression(penalty=None, fit_intercept=True).fit(logit_probabilities, ground_truth)

    return lr.coef_.item()

# Comparing loss on individual points
def get_fraction_similar_error(X, y, X_second, first_model, second_model):
    y_pred_first = first_model.predict(X)
    y_pred_second = second_model.predict(X_second)
    if hasattr(first_model, 'predict_proba'):
        y_pred_prob_first = first_model.predict_proba(X)
    else:
        print('predict_proba not defined for', first_model)
        y_pred_prob_first = y_pred_first  # use 0/1 predictions instead of predicted probabilities for Linear SVC
    if hasattr(second_model, 'predict_proba'):
        y_pred_prob_second = second_model.predict_proba(X_second)
    else:
        print('predict_proba not defined for', second_model)
        y_pred_prob_second = y_pred_second
    logloss_sample_first = per_sample_log_loss(y, y_pred_prob_first)
    logloss_sample_second = per_sample_log_loss(y, y_pred_prob_second)
    similar_errors = np.isclose(logloss_sample_first, logloss_sample_second, rtol=1e-05, atol=1e-02)
    return np.mean(similar_errors)

def per_sample_log_loss(y_true, y_pred, eps=1e-15):
    """
    Log loss for each sample.
    Simplified version of log_loss implemented in sklearn
    """
    # Convert y_true to 0/1 label
    lb = LabelBinarizer()
    y_true = lb.fit_transform(y_true)
    if y_true.shape[1]==1:
        y_true = np.append(1-y_true, y_true, axis=1)

    y_pred = np.clip(y_pred, a_min=eps, a_max=1-eps)  # avoids taking log of 0 or inf

    # Make y_pred of shape (nsamples,nclasses)
    if len(y_pred.shape)==1:
        y_pred = y_pred[:, np.newaxis]
    if y_pred.shape[1]==1:
        y_pred = np.append(1-y_pred, y_pred, axis=1)

    loss = -1 * (y_true * np.log(y_pred)).sum(axis=1)

    return loss