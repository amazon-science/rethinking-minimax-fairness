# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Original code from this repo on GitHub

minimax-fair
    Repo URL https://github.com/amazon-science/minimax-fair
    Written by gillwesl-aws (https://github.com/gillwesl) and jimmyren23 (Jimmy Ren, https://github.com/jimmyren23)
    Distributed under Apache 2.0 license
    Code URL https://github.com/amazon-science/minimax-fair/blob/392af13d909f28495cec375374e5884c567d083b/dataset_mapping.py

Instructions to download the datasets are at
https://github.com/amazon-science/minimax-fair#datasets

No modification.
"""

def get_dataset_features(dataset):
    """
    Takes in the name of a dataset and returns the relevant parameters
    """
    groups_to_drop = []
    if dataset == 'COMPAS':
        path = 'Datasets/compas-scores-two-years.csv'
        label = 'two_year_recid'  # binary
        group = 'race'  # 'sex' also an option
        usable_features = ['race', 'age', 'sex', 'priors_count', 'c_charge_degree', 'juv_fel_count']
        categorical_columns = []
        groups_to_drop = ['race@Asian', 'race@Native American']
        is_categorical = True
    elif dataset == 'COMPAS_full':
        path = 'Datasets/compas-scores-two-years.csv'
        label = 'two_year_recid'  # binary
        group = 'race'  # 'sex' also an option
        usable_features = ['race', 'age', 'sex', 'priors_count', 'c_charge_degree', 'juv_fel_count']
        categorical_columns = []
        is_categorical = True
    elif dataset == 'COMPAS_race_and_gender':
        path = 'Datasets/compas-scores-two-years.csv'
        label = 'two_year_recid'  # binary
        group = ['race', 'sex']
        usable_features = ['race', 'age', 'sex', 'priors_count', 'c_charge_degree', 'juv_fel_count']
        categorical_columns = []
        groups_to_drop = ['race@Asian', 'race@Native American']
        is_categorical = True
    elif dataset == 'Default':
        path = 'Datasets/default.csv'
        label = 'default payment next month'  # binary
        group = 'SEX'
        usable_features = None  # all
        categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE']
        is_categorical = True
    elif dataset == 'Communities':
        path = 'Datasets/communities_cleaned.csv'
        label = 'ViolentCrimesPerPop'
        group = 'pluralityRace'
        usable_features = None  # all
        categorical_columns = []
        is_categorical = False
    elif dataset == 'Adult':
        path = 'Datasets/adult_cleaned.csv'
        label = 'income'  # binary <50k or not
        group = 'race'
        usable_features = None  # all
        categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                               'race', 'sex', 'native-country']
        is_categorical = True
    elif dataset == 'Student':
        path = 'Datasets/student-mat.csv'
        label = 'G3'  # binary
        group = 'sex'  # TBD
        usable_features = None  # all
        categorical_columns = ['Medu', 'Fedu', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absenses']
        is_categorical = False
    elif dataset == 'Bike':
        path = 'Datasets/SeoulBikeData.csv'
        label = 'Rented Bike Count'
        group = 'Seasons'
        usable_features = ['Seasons', 'Hour', 'Temperature(C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)',
                           'Dew point temperature(C)', 'Solar Radiation (MJ/m2)',
                           'Rainfall(mm)', 'Snowfall (cm)', 'Seasons', 'Holiday', 'Functioning Day']
        categorical_columns = ['Seasons', 'Holiday', 'Functioning']
        is_categorical = False
    elif dataset == 'Credit':
        path = 'Datasets/german_cleaned.csv'
        label = 'Creditability'
        group = 'Sex & Marital Status'
        usable_features = None  # all
        categorical_columns = []
        is_categorical = True 
    elif dataset == 'Fires':
        path = 'Datasets/forestfires.csv'
        label = 'area'
        group = 'month'
        usable_features = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp',
                           'RH', 'wind', 'rain', 'area']
        categorical_columns = ['day', 'month']  # non-linear (but monotone) numerical columns on scale such as 1-4
        is_categorical = False
    elif dataset == 'Heart':
        path = 'Datasets/heart_failure_clinical_records_dataset.csv'
        label = 'DEATH_EVENT'
        group = 'sex'
        usable_features = None
        categorical_columns = []
        is_categorical = False
    elif dataset == 'Marketing(Full)':
        path = 'Datasets/bank-full.csv'
        label = 'y'
        group = 'job'
        usable_features = None
        categorical_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
        is_categorical = True
    elif dataset == 'Marketing(Small)':
        path = 'Datasets/bank.csv'
        label = 'y'
        group = 'job'
        usable_features = None
        categorical_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
        is_categorical = True
    elif dataset == 'Wine':
        path = 'Datasets/winequality-full.csv'
        label = 'quality'
        group = 'color'
        usable_features = None
        categorical_columns = ['color']
        is_categorical = False
    # Use synthetic data if any other string is passed in
    else:
        path = ''
        label = None
        group = None
        usable_features = None
        categorical_columns = []
        is_categorical = True
    return path, label, group, usable_features, categorical_columns, groups_to_drop, is_categorical
