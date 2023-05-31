# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


#!/usr/bin/env python
# coding: utf-8

"""
Code for reading datasets

Credit
------

Original code for preparing Adult Income data is from the Autogluon tutorial
    URL https://auto.gluon.ai/stable/tutorials/tabular_prediction/tabular-custom-model.html
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  # resolves libomp error thrown by xgboost after installing autogluon
import numpy as np
import pandas as pd
import torch

from folktables_helper import ACSEmploymentSEX, ACSEmploymentRACE, ACSEmploymentSEXGroupRemoved, ACSEmploymentRACEGroupRemoved
from folktables_helper import ACSIncomeSEX, ACSIncomeRACE, ACSIncomeSEXGroupRemoved, ACSIncomeRACEGroupRemoved
from folktables_helper import ACSPublicCoverageSEX, ACSPublicCoverageRACE, ACSPublicCoverageSEXGroupRemoved, ACSPublicCoverageRACEGroupRemoved
from folktables_helper import categories_limited
from folktables import ACSDataSource
from autogluon.tabular import TabularDataset
from autogluon.core.data import LabelCleaner
from autogluon.core.utils import infer_problem_type
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

from prepare_datasets import prepare_COMPAS, prepare_diabetes, prepare_drug_consumption
from prepare_datasets import prepare_eicu, prepare_datasets_minimax_paper

NOT_CATEGORICAL_FEATURES = ['AGEP', 'SCHL', 'RELP']
 # Edit to include more states in their USPS Code format in 
 # https://www.census.gov/library/reference/code-lists/ansi/ansi-codes-for-states.html 
 # Expand FIPS Codes for the States and District of Columbia
US_STATES_FOLKTABLES = ['NY', 'CA', 'TX', 'IN']
DATASETS_MINIMAX_PAPER = [
    ('Default', 'SEX'),
    ('Communities', 'pluralityRace'),
    ('Credit', 'Sex & Marital Status'),
    ('Heart', 'sex'),
    ('Marketing(Full)', 'job'),
]

def race_coding_adult_income_uci(data):
    '''
    Recode all data points with race not equal to either White, Black or African American, Asian into a single code
    4. White
    2. Black
    5. Other = (1. Asian-Pac-Islander, 0. Amer-Indian-Eskimo, 3. Other)
    White                 33416
    Black                  3794
    Asian-Pac-Islander     1191
    Amer-Indian-Eskimo      347
    Other                   325
    '''
    data['race_prev_code'] = data['race']
    data.loc[~data['race_prev_code'].isin([' Black',' White']), 'race'] = ' Other'
    return data

def prepare_adult_data_autogluon(group_type):
    """
    Code for preparing data taken from Autogluon tutorial
    https://auto.gluon.ai/0.4.0/tutorials/tabular_prediction/tabular-custom-model.html
    """
    train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')  # can be local CSV file as well, returns Pandas DataFrame
    test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')  # another Pandas DataFrame
    # Recode race variables into 3 levels
    if group_type == 'race':
        train_data = race_coding_adult_income_uci(train_data)
        test_data = race_coding_adult_income_uci(test_data)
    label = 'class'
    X = train_data.drop(columns=[label])
    y = train_data[label]
    X_test = test_data.drop(columns=[label])
    y_test = test_data[label]
    # Construct a LabelCleaner to neatly convert labels to float/integers during model training/inference, can also use to inverse_transform back to original.
    problem_type = infer_problem_type(y=y)  # Infer problem type (or else specify directly)
    label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
    y_train_clean = label_cleaner.transform(y)

    print(f'Labels cleaned: {label_cleaner.inv_map}')
    print(f'inferred problem type as: {problem_type}')

    feature_generator = AutoMLPipelineFeatureGenerator()
    X_train_clean = feature_generator.fit_transform(X)
    X_test_clean = feature_generator.transform(X_test)
    y_test_clean = label_cleaner.transform(y_test)

    group_column = group_type
    group_train = train_data[group_column]
    group_test = test_data[group_column]

    torch.save((X_train_clean, y_train_clean, group_train), f'Datasets/adult_income_uci/train_clean_{group_type}.pt')
    torch.save((X_test_clean, y_test_clean, group_test), f'Datasets/adult_income_uci/test_clean_{group_type}.pt')

class Dataset:
    '''
    Reads ACS datasets using folktables functions, optionally converts categorical variables to dummies, and downsamples data
    '''
    def __init__(self) -> None:
        _us_states = US_STATES_FOLKTABLES
        _folktables_dataset_types = ['adult_income', 'adult_employment', 'adult_health_insurance']
        self._list_of_datasets = [(f'{d}_{s}','sex') for d in _folktables_dataset_types for s in _us_states]\
                                + [(f'{d}_{s}','race') for d in _folktables_dataset_types for s in _us_states]\
                                + [('adult_income_uci','sex'), ('adult_income_uci','race')]\
                                + [('compas','race'), ('diabetes','age')]\
                                + [('drugconsumption','country')]\
                                + [('eicu','sex'), ('eicu','race')]\
                                + DATASETS_MINIMAX_PAPER  # format (adult_income_NY, sex), (adult_employment_NY, race)

    def list_datasets(self):
        return self._list_of_datasets

    def read_dataset(self, args, dataset_name, group_type):
        if dataset_name == 'adult_income_uci':
            if group_type in ['sex', 'race']:
                if (not os.path.exists(f'./Datasets/adult_income_uci/train_clean_{group_type}.pt')) or (not os.path.exists(f'./Datasets/adult_income_uci/test_clean_{group_type}.pt')):
                    prepare_adult_data_autogluon(group_type)
                (X_train_clean, y_train_clean, group_train) = torch.load(f'Datasets/adult_income_uci/train_clean_{group_type}.pt')
            else:
                raise NotImplementedError(f'{group_type} for the dataset {dataset_name}')
            if group_type == 'sex':
                X_train_clean_group_removed = X_train_clean.drop(columns=['sex'])
            elif group_type == 'race':
                X_train_clean_group_removed = X_train_clean.drop(columns=['race', 'race_prev_code'])
            features, label, group = X_train_clean.to_numpy(), y_train_clean.to_numpy(), group_train.to_numpy()
            features_group_removed = X_train_clean_group_removed.to_numpy()


        elif ('adult_income' in dataset_name) or ('adult_employment' in dataset_name) or ('adult_health_insurance' in dataset_name):
            if 'adult_income' in dataset_name:
                state_name = dataset_name.replace('adult_income_','')
                ACSObjectSEX, ACSObjectSEXGroupRemoved = ACSIncomeSEX, ACSIncomeSEXGroupRemoved
                ACSObjectRACE, ACSObjectRACEGroupRemoved = ACSIncomeRACE, ACSIncomeRACEGroupRemoved
            elif 'adult_employment' in dataset_name:
                state_name = dataset_name.replace('adult_employment_','')
                ACSObjectSEX, ACSObjectSEXGroupRemoved = ACSEmploymentSEX, ACSEmploymentSEXGroupRemoved
                ACSObjectRACE, ACSObjectRACEGroupRemoved = ACSEmploymentRACE, ACSEmploymentRACEGroupRemoved
            elif 'adult_health_insurance' in dataset_name:
                state_name = dataset_name.replace('adult_health_insurance_','')
                ACSObjectSEX, ACSObjectSEXGroupRemoved = ACSPublicCoverageSEX, ACSPublicCoverageSEXGroupRemoved
                ACSObjectRACE, ACSObjectRACEGroupRemoved = ACSPublicCoverageRACE, ACSPublicCoverageRACEGroupRemoved
            data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person', root_dir='Datasets')
            acs_data = data_source.get_data(states=[state_name], download=True)

            categories = categories_limited
            if group_type == 'sex':
                features, label, group = ACSObjectSEX.df_to_pandas(
                                            acs_data, 
                                            categories=categories if args.to_categorical else None, 
                                            dummies=args.to_categorical
                                        )
                features_group_removed, _, _ = ACSObjectSEXGroupRemoved.df_to_pandas(
                                                    acs_data,
                                                    categories=categories if args.to_categorical else None, 
                                                    dummies=args.to_categorical
                                                )
            elif group_type == 'race':
                features, label, group = ACSObjectRACE.df_to_pandas(
                                            acs_data, 
                                            categories=categories if args.to_categorical else None, 
                                            dummies=args.to_categorical
                                        )
                features_group_removed, _, _ = ACSObjectRACEGroupRemoved.df_to_pandas(
                                                    acs_data, 
                                                    categories=categories if args.to_categorical else None, 
                                                    dummies=args.to_categorical
                                                )
            else:
                raise NotImplementedError(f"grouping data on {group_type}")
            features, label, group = features.to_numpy(), label.to_numpy().flatten(), group.to_numpy().flatten()
            features_group_removed = features_group_removed.to_numpy()


        elif dataset_name == 'compas':
            if group_type == 'race':
                features_group_removed, label, group, group_names = prepare_COMPAS()
            else:
                raise NotImplementedError(f"grouping data on {group_type}")
            features = np.concatenate([features_group_removed, group[:,np.newaxis]], axis=1)


        elif dataset_name == 'diabetes':
            if group_type == 'age':
                features_group_removed, label, group, group_names = prepare_diabetes()
            else:
                raise NotImplementedError(f"grouping data on {group_type}")
            features = np.concatenate([features_group_removed, group[:,np.newaxis]], axis=1)


        elif dataset_name == 'drugconsumption':
            if group_type == 'country':
                features_group_removed, label, group, group_names = prepare_drug_consumption()
            else:
                raise NotImplementedError(f"grouping data on {group_type}")
            features = np.concatenate([features_group_removed, group[:,np.newaxis]], axis=1)

        elif dataset_name == 'eicu':
            features_group_removed, label, group, group_names = prepare_eicu(group_type)
            features = np.concatenate([features_group_removed, group[:,np.newaxis]], axis=1)

        elif (dataset_name, group_type) in DATASETS_MINIMAX_PAPER:
            features, features_group_removed, label, group, _, group_names = prepare_datasets_minimax_paper(dataset_name)
            group = group.squeeze()  # [[0,1,...]] to [0,1,...]

        else:
            raise NotImplementedError(f'dataset handler not found {dataset_name}')

        # Downsample rows of dataset to run faster
        if features.shape[0] > args.data_max_size:
            rng = np.random.default_rng(seed=1)
            indices = np.arange(features.shape[0])
            random_indices = rng.permutation(indices)[:args.data_max_size]
            features = features[random_indices]
            label = label[random_indices]
            group = group[random_indices]
            features_group_removed = features_group_removed[random_indices]

        if not (np.std(features_group_removed, axis=0)>0).all():
            print("A column in features_group_removed has zero std dev")

        # Map group names to indices from 0 to number of groups-1
        group_label_map = dict([(g,i) for i,g in enumerate(sorted(set(group)))])
        group = np.array([group_label_map[g] for g in group])
        print(f"Mapping labels to {group_label_map}")

        group_names = list(group_label_map.keys())
        
        print(f'Data read X={features.shape}, y={label.shape}, group={group.shape}, X_group_removed={features_group_removed.shape}')
        print(f'Sample features\n', features[:5])
        print(f'Sample labels\n', label[:5])
        print(f'Sample group\n', group[:5])
        return features, label, group, features_group_removed, group_names, group_label_map