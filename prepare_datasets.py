# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Code to read datasets

Credit
------

Original code from the following two repos on GitHub

active-sampling-for-minmax-fairness
    Repo URL https://github.com/amazon-science/active-sampling-for-minmax-fairness
    Written by Matthäus Kleindessner
    Distributed under Apache 2.0 license
    Code URL https://github.com/amazon-science/active-sampling-for-minmax-fairness/blob/02617eb299fc34de3851b3c88dfcf9d388ca8970/prepare_datasets.py

minimax-fair
    Repo URL https://github.com/amazon-science/minimax-fair
    Written by gillwesl-aws and jimmyren23 (Jimmy Ren)
    Distributed under Apache 2.0 license
    Code URL https://github.com/amazon-science/minimax-fair/blob/a237326f10bb752f6d6c8e5e9e4bb6fc849b3427/main_driver.py

Instructions to download the datasets are at
https://github.com/amazon-science/minimax-fair#datasets
"""

import numpy as np
import os
import pandas
import pathlib
import requests
from sklearn.feature_selection import VarianceThreshold
import sys
import zipfile
from src.setup_matrices import setup_matrices
from src.read_file import read_dataset_from_file
from dataset_mapping import get_dataset_features


def prepare_drug_consumption():
    # Download data file if it does not exist
    if (not os.path.exists('Datasets/drug_consumption.data')):
        print('Data set does not exist in current folder --- have to download it')
        pathlib.Path('Datasets').mkdir(exist_ok=True)
        r = requests.get(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data',
            allow_redirects=True)
        if r.status_code == requests.codes.ok:
            print('Download successful\n')
        else:
            print('Could not download the data set --- please download it manually')
            sys.exit()
        open('Datasets/drug_consumption.data', 'wb').write(r.content)

    data = pandas.read_csv('Datasets/drug_consumption.data',
                           names=['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity',
                                  'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive',
                                  'SS', 'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis',
                                  'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine',
                                  'Legalh', 'LSD', 'Meth', 'Mushroom', 'Nicotine', 'Semer', 'VSA'])

    features_for_predicting_Y = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive',
                                 'SS']
    X = data.loc[:, features_for_predicting_Y].values.astype('float64')

    data['Cannabis'].replace(
        to_replace={'CL0': 0, 'CL1': 0, 'CL2': 1, 'CL3': 1, 'CL4': 1, 'CL5': 1, 'CL6': 1},
        inplace=True)
    label = data['Cannabis'].values.astype('int64')

    groups = ['US', 'UK', 'Other']
    protected_attribute = np.zeros(len(label), dtype=int)  # NOTE: use one-hot vector
    protected_attribute[np.isclose(data['Country'].values, -0.57009)] = 1
    protected_attribute[np.isclose(data['Country'].values, 0.96082)] = 2
    protected_attribute[protected_attribute == 0] = 3
    protected_attribute -= 1

    return X, label, protected_attribute, groups


def prepare_COMPAS():
    # Download data file if it does not exist
    if (not os.path.exists('Datasets/compas-scores-two-years.csv')):
        print('Data set does not exist in current folder --- have to download it')
        pathlib.Path('Datasets').mkdir(exist_ok=True)
        r = requests.get(
            'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv',
            allow_redirects=True)
        if r.status_code == requests.codes.ok:
            print('Download successful\n')
        else:
            print('Could not download the data set --- please download it manually')
            sys.exit()
        open('Datasets/compas-scores-two-years.csv', 'wb').write(r.content)

    data = pandas.read_csv('Datasets/compas-scores-two-years.csv', header=0,
                           usecols=['race', 'age', 'sex', 'priors_count', 'c_charge_degree',
                                    'juv_fel_count', 'two_year_recid'])

    data['sex'] = data['sex'].astype('category')
    data['c_charge_degree'] = data['c_charge_degree'].astype('category')
    features_for_predicting_Y = ['age', 'sex', 'priors_count', 'c_charge_degree', 'juv_fel_count']
    X = pandas.get_dummies(data.loc[:, features_for_predicting_Y],
                           drop_first=True).values.astype('float64')

    label = data['two_year_recid'].values.astype('int64')

    groups = ['African-American', 'Caucasian', 'Hispanic', 'Other']
    protected_attribute = 3 * np.ones(len(label), dtype=int)  # use one-hot vector
    for counter, gr in enumerate(groups[:-1]):
        protected_attribute[data['race'] == gr] = counter

    return X, label, protected_attribute, groups


def prepare_diabetes():
    # Download data file if it does not exist
    if (not os.path.exists('Datasets/dataset_diabetes/diabetic_data.csv')):
        print('Data set does not exist in current folder --- have to download it')
        pathlib.Path('Datasets').mkdir(exist_ok=True)
        r = requests.get(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip',
            allow_redirects=True)
        if r.status_code == requests.codes.ok:
            print('Download successful\n')
        else:
            print('Could not download the data set --- please download it manually')
            sys.exit()
        open('Datasets/dataset_diabetes.zip', 'wb').write(r.content)
        with zipfile.ZipFile('Datasets/dataset_diabetes.zip', 'r') as zip_ref:
            zip_ref.extractall('Datasets')

    data = pandas.read_csv('Datasets/dataset_diabetes/diabetic_data.csv', header=0, na_values='?',
                           usecols=['gender', 'age',
                                    'admission_type_id',
                                    'time_in_hospital', 'num_lab_procedures',
                                    'num_procedures', 'num_medications', 'number_outpatient',
                                    'number_emergency', 'number_inpatient', 'number_diagnoses',
                                    'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed',
                                    'readmitted'])

    data = data.dropna(axis=0)
    for feature in ['gender', 'admission_type_id', 'max_glu_serum', 'A1Cresult', 'change',
                    'diabetesMed']:
        data[feature] = data[feature].astype('category')

    groups_fine = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)',
                   '[70-80)', '[80-90)', '[90-100)']
    groups_encoding = [0, 0, 0, 0, 0, 1, 2, 3, 4, 4]
    groups = ['[0-50)', '[50-60)', '[60-70)', '[70-80)', '[80-100)']

    protected_attribute = np.zeros(data.shape[0], dtype=int)  # NOTE: use one-hot vector
    for ell, gr in enumerate(groups_fine):
        protected_attribute[data['age'] == gr] = groups_encoding[ell]

    label = np.zeros(data.shape[0], dtype=int)
    label[data['readmitted'] == 'No'] = 1
    label[data['readmitted'] == '>30'] = 1

    data.drop(labels=['age', 'readmitted'], axis=1, inplace=True)
    X = pandas.get_dummies(data, drop_first=True).values.astype('float64')

    # remove low-variance features
    selector = VarianceThreshold(0.001)
    X = selector.fit_transform(X)

    return X, label, protected_attribute, groups

def prepare_eicu(group_type):
    data = pandas.read_csv('Datasets/eicu_day1_saps2_without_imputation.csv', sep=',', index_col=0)
    data = data.sample(n=20000, axis=0, random_state=0)  # subsample to save runtime
    if group_type=='sex':
        sensitive_feature = 'is_female'
        groups = ['Not Female', 'Female']
    elif group_type=='race':
        sensitive_feature = 'race_other'
        groups = ['Not AAHispAsian', 'AAHispAsian']
        # definition - race_other=1 if ethnicity not in 'African American', 'Hispanic', 'Asian'
        # from https://github.com/alistairewj/icu-model-transfer/blob/master/datasets/eicu/static-data.sql
        # ethnicity - Asian, Caucasian, African American, Native American, Hispanic, Other/Unknown
    else:
        raise NotImplementedError(f"grouping data on {group_type}")
    
    target = 'death'
    features = ['heartrate',
       'sysbp',
       'temp',
       'bg_pao2fio2ratio',
       'bun',
       'urineoutput',
       'sodium',
       'potassium',
       'bicarbonate',
       'bilirubin',
       'wbc',
       'gcs',
       'age',
       'electivesurgery']

    label = data[target].values
    protected_attribute = data[sensitive_feature].values

    # Add missing value indicators for each feature
    data_indicators = data[features].isnull().astype(int).add_suffix('_indicator')
    data_fill_0 = data[features].fillna(0)
    X = pandas.concat([data_fill_0, data_indicators], axis=1).values
    
    # remove low-variance features
    selector = VarianceThreshold(0.001)
    X = selector.fit_transform(X)

    return X, label, protected_attribute, groups

def prepare_datasets_minimax_paper(dataset_name):
    save_data = False  # Whether or not data from setting up matrices should be is saved to the specified directory
    file_dir = 'vectorized_datasets'  # Directory for files containing vectorized datasets to read from/write to
    file_name = '<INSERT NAME HERE>.npz'  # File name within file_dir from which to read or write data, should be .npz file
    
    path, label, groups, usable_features, categorical_columns, groups_to_drop, _ \
            = get_dataset_features(dataset_name)
    
    drop_group_as_feature = True  # Set to False (default) if groups should also be a one hot encoded categorical feature
    features_group_removed, _, _, _, _, _ = \
            setup_matrices(path, label, groups, usable_features=usable_features,
                           drop_group_as_feature=drop_group_as_feature,
                           categorical_columns=categorical_columns, groups_to_drop=groups_to_drop,
                           verbose=False,
                           save_data=save_data, file_dir=file_dir, file_name=file_name)
    
    drop_group_as_feature = False
    verbose = False
    features, label, grouplabels, group_names, _, is_binary = \
            setup_matrices(path, label, groups, usable_features=usable_features,
                           drop_group_as_feature=drop_group_as_feature,
                           categorical_columns=categorical_columns, groups_to_drop=groups_to_drop,
                           verbose=verbose,
                           save_data=save_data, file_dir=file_dir, file_name=file_name)
    
    assert is_binary, 'label takes more than two values'
    
    features = features.astype(float)
    features_group_removed = features_group_removed.astype(float)
    
    return features, features_group_removed, label, grouplabels, groups, group_names



