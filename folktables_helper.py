# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


#!/usr/bin/env python
# coding: utf-8

"""
Helper functions to read folktables data

Credit
------
Functions are modified from their implementation in folktables package
Released with an MIT License
features and target in class ACSDataSource are modified
Repo URL
    https://github.com/socialfoundations/folktables
File URL
    https://github.com/socialfoundations/folktables/blob/5ff4ad7d7f67f0fdf71c8a014e4756e5b64c7f0c/folktables/acs.py
Authors
    https://github.com/jenno-verdonck
    https://github.com/millerjohnp
    https://github.com/mrtzh
"""
import numpy as np
import pandas as pd

from folktables import BasicProblem

"""
Encodings:
SEX 
1. Male 
2. Female

RAC1P
1 .White alone
2 .Black or African American alone
3 .American Indian alone
4 .Alaska Native alone
5 .American Indian and Alaska Native tribes specified; or
.American Indian or Alaska Native, not specified and no other
.races
6 .Asian alone
7 .Native Hawaiian and Other Pacific Islander alone
8 .Some Other Race alone
9 .Two or More Races

Age, Educational attaintment, Marital status, Retirement income, Disability, Employment status of parents, Citizenship status, Mobility status,
Military service, Ancestry recode, Nativity, Hearing difficulty, Vision difficulty, Cognitive difficulty, Sex, Recoded detailed race code
"""

def adult_filter(data):
    """Mimic the filters in place for Adult data.
    Adult documentation notes: Extraction was done by Barry Becker from
    the 1994 Census database. A set of reasonably clean records was extracted
    using the following conditions:
    ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    """
    df = data
    df = df[df['AGEP'] > 16]
    df = df[df['PINCP'] > 100]
    df = df[df['WKHP'] > 0]
    df = df[df['PWGTP'] >= 1]
    return df

# categorical variables OCCP, POBP kept as numerical since it has 531, 225 categories
ACSIncomeSEX = BasicProblem(
    features=[
        'AGEP',
        'COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',
        'RAC1P',
    ],
    target='PINCP',
    target_transform=lambda x: x > 50000,
    group='SEX',
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

ACSIncomeSEXGroupRemoved = BasicProblem(
    features=[
        'AGEP',
        'COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'RAC1P',
    ],
    target='PINCP',
    target_transform=lambda x: x > 50000,
    group='SEX',
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

ACSIncomeRACE = BasicProblem(
    features=[
        'AGEP',
        'COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',
        'RAC1P',
        'RAC1PR',
    ],
    target='PINCP',
    target_transform=lambda x: x > 50000,
    group='RAC1PR',
    preprocess=lambda df: race_coding(adult_filter(df)),
    postprocess=lambda x: np.nan_to_num(x, -1),
)

ACSIncomeRACEGroupRemoved = BasicProblem(
    features=[
        'AGEP',
        'COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',
    ],
    target='PINCP',
    target_transform=lambda x: x > 50000,
    group='RAC1PR',
    preprocess=lambda df: race_coding(adult_filter(df)),
    postprocess=lambda x: np.nan_to_num(x, -1),
)

ACSEmploymentSEX = BasicProblem(
    features=[
        'AGEP',
        'SCHL',
        'MAR',
        'RELP',
        'DIS',
        'ESP',
        'CIT',
        'MIG',
        'MIL',
        'ANC',
        'NATIVITY',
        'DEAR',
        'DEYE',
        'DREM',
        'SEX',
        'RAC1P',
    ],
    target='ESR',
    target_transform=lambda x: x == 1,
    group='SEX',
    preprocess=lambda x: x,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

ACSEmploymentSEXGroupRemoved = BasicProblem(
    features=[
        'AGEP',
        'SCHL',
        'MAR',
        'RELP',
        'DIS',
        'ESP',
        'CIT',
        'MIG',
        'MIL',
        'ANC',
        'NATIVITY',
        'DEAR',
        'DEYE',
        'DREM',
        'RAC1P',
    ],
    target='ESR',
    target_transform=lambda x: x == 1,
    group='SEX',
    preprocess=lambda x: x,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

def race_coding(data):
    '''
    Recode all data points with race not equal to either White, Black or African American, Asian into a single code
    1 .White alone
    2 .Black or African American alone
    6 .Asian alone
    10. Other
    '''
    data['RAC1PR'] = data['RAC1P']
    data.loc[~data['RAC1P'].isin([1,2,6]), 'RAC1PR'] = 10
    return data

ACSEmploymentRACE = BasicProblem(
    features=[
        'AGEP',
        'SCHL',
        'MAR',
        'RELP',
        'DIS',
        'ESP',
        'CIT',
        'MIG',
        'MIL',
        'ANC',
        'NATIVITY',
        'DEAR',
        'DEYE',
        'DREM',
        'SEX',
        'RAC1P',
        'RAC1PR',
    ],
    target='ESR',
    target_transform=lambda x: x == 1,
    group='RAC1PR',
    preprocess=race_coding,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

ACSEmploymentRACEGroupRemoved = BasicProblem(
    features=[
        'AGEP',
        'SCHL',
        'MAR',
        'RELP',
        'DIS',
        'ESP',
        'CIT',
        'MIG',
        'MIL',
        'ANC',
        'NATIVITY',
        'DEAR',
        'DEYE',
        'DREM',
        'SEX',
    ],
    target='ESR',
    target_transform=lambda x: x == 1,
    group='RAC1PR',
    preprocess=race_coding,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

def public_coverage_filter(data):
    """
    Filters for the public health insurance prediction task; focus on low income Americans, and those not eligible for Medicare
    """
    df = data
    df = df[df['AGEP'] < 65]
    df = df[df['PINCP'] <= 30000]
    return df

ACSPublicCoverageSEX = BasicProblem(
    features=[
        'AGEP',
        'SCHL',
        'MAR',
        'SEX',
        'DIS',
        'ESP',
        'CIT',
        'MIG',
        'MIL',
        'ANC',
        'NATIVITY',
        'DEAR',
        'DEYE',
        'DREM',
        'PINCP',
        'ESR',
        'FER',
        'RAC1P',
    ],
    target='PUBCOV',
    target_transform=lambda x: x == 1,
    group='SEX',
    preprocess=public_coverage_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

ACSPublicCoverageSEXGroupRemoved = BasicProblem(
    features=[
        'AGEP',
        'SCHL',
        'MAR',
        'DIS',
        'ESP',
        'CIT',
        'MIG',
        'MIL',
        'ANC',
        'NATIVITY',
        'DEAR',
        'DEYE',
        'DREM',
        'PINCP',
        'ESR',
        'FER',
        'RAC1P',
    ],
    target='PUBCOV',
    target_transform=lambda x: x == 1,
    group='SEX',
    preprocess=public_coverage_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

ACSPublicCoverageRACE = BasicProblem(
    features=[
        'AGEP',
        'SCHL',
        'MAR',
        'SEX',
        'DIS',
        'ESP',
        'CIT',
        'MIG',
        'MIL',
        'ANC',
        'NATIVITY',
        'DEAR',
        'DEYE',
        'DREM',
        'PINCP',
        'ESR',
        'FER',
        'RAC1P',
        'RAC1PR',
    ],
    target='PUBCOV',
    target_transform=lambda x: x == 1,
    group='RAC1PR',
    preprocess=lambda df: race_coding(public_coverage_filter(df)),
    postprocess=lambda x: np.nan_to_num(x, -1),
)

ACSPublicCoverageRACEGroupRemoved = BasicProblem(
    features=[
        'AGEP',
        'SCHL',
        'MAR',
        'SEX',
        'DIS',
        'ESP',
        'CIT',
        'MIG',
        'MIL',
        'ANC',
        'NATIVITY',
        'DEAR',
        'DEYE',
        'DREM',
        'PINCP',
        'ESR',
        'FER',
    ],
    target='PUBCOV',
    target_transform=lambda x: x == 1,
    group='RAC1PR',
    preprocess=lambda df: race_coding(public_coverage_filter(df)),
    postprocess=lambda x: np.nan_to_num(x, -1),
)

# This dictionary is created from the data dictionary available at the US Census website
# https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.csv
# It combines levels that are assigned to less than 10% of rows into separate categories
categories_limited = {
    'SCHL': {
        16.0: 'Regular high school diploma',
        19.0: '1 or more years of college credit, no degree',
        21.0: "Bachelor's degree",
        18.0: "Associate's degree",
        20.0: "Other College",
        22.0: "Other College",
        23.0: "Other College",
        24.0: 'Other College',
        1.0: 'Other',
        2.0: 'Other',
        3.0: 'Other',
        4.0: 'Other',
        5.0: 'Other',
        6.0: 'Other',
        7.0: 'Other',
        8.0: 'Other',
        9.0: 'Other',
        10.0: 'Other',
        11.0: 'Other',
        12.0: 'Other',
        13.0: 'Other',
        14.0: 'Other',
        15.0: 'Other',
        17.0: 'Other',
        float('nan'): 'Other'
    },
    'MAR': {
        1: 'Married',
        5: 'Never married or under 15 years old',
        2: 'Other',
        3: 'Other',
        4: 'Other',
        float('nan'): 'Other'
    },
    'SEX': {1: 'Male', 2: 'Female', float('nan'): 'N/A'},
    'DIS': {1: 'With a disability', 2: 'Without a disability', float('nan'): 'N/A'},
    'ESP': {
            float('nan'): 'N/A (not own child of householder, and not child in subfamily) Living with two parents:',
            1.0: 'Other',
            2.0: 'Other',
            3.0: 'Other',
            4.0: 'Other',
            5.0: 'Other',
            6.0: 'Other',
            7.0: 'Other',
            8.0: 'Other',
    },
    'CIT': {1: 'Born in the U.S.',
            2: 'Other',
            3: 'Other',
            4: 'U.S. citizen by naturalization',
            5: 'Not a citizen of the U.S.',
            float('nan'): 'Other'},
    'MIG': {1.0: 'Yes, same house (nonmovers)',
            2.0: 'Other',
            3.0: 'No, different house in US or Puerto Rico',
            float('nan'): 'Other'},
    'MIL': {1.0: 'Other',
            2.0: 'Other',
            3.0: 'Other',
            4.0: 'Never served in the military',
            float('nan'): 'Other'},
    'ANC': {1: 'Single',
            2: 'Multiple',
            3: 'Other',
            4: 'Not reported',
            8: 'Other',
            float('nan'): 'Other'},
    'NATIVITY': {1: 'Native', 2: 'Foreign born', float('nan'): 'N/A'},
    'DEAR': {1: 'Yes', 2: 'No', float('nan'): 'N/A'},
    'DEYE': {1: 'Yes', 2: 'No', float('nan'): 'N/A'},
    'DREM': {1.0: 'Yes', 2.0: 'No', float('nan'): 'N/A (Less than 5 years old)'},
    'ESR': {1.0: 'Civilian employed, at work',
            2.0: 'Other',
            3.0: 'Unemployed',
            4.0: 'Other',
            5.0: 'Other',
            6.0: 'Not in labor force',
            float('nan'): 'Other'},
    'FER': {1.0: 'Yes',
            2.0: 'No',
            float('nan'): 'N/A (less than 15 years/greater than 50 years/ male)'},
    'COW': {1.0: 'Employee of a private for-profit company or business, or of an individual, for wages, salary, or commissions',
            2.0: 'Employee of a private not-for-profit, tax-exempt, or charitable organization',
            3.0: 'Local government employee (city, county, etc.)',
            4.0: 'Other',
            5.0: 'Other',
            6.0: 'Other',
            7.0: 'Other',
            8.0: 'Other',
            9.0: 'Other',
            float('nan'): 'Other'},
    'RELP': {0: 'Reference person',
            1: 'Husband/wife',
            2: 'Biological son or daughter',
            3: 'Other',
            4: 'Other',
            5: 'Other',
            6: 'Other',
            7: 'Other',
            8: 'Other',
            9: 'Other',
            10: 'Other',
            11: 'Other',
            12: 'Other',
            13: 'Other',
            14: 'Other',
            15: 'Other',
            16: 'Other',
            17: 'Other',
            float('nan'): 'Other'},
    'RAC1P': {1: 'White alone',
            2: 'Black or African American alone',
            3: 'American Indian alone',
            4: 'Alaska Native alone',
            5: 'American Indian and Alaska Native tribes specified; or American Indian or Alaska Native, not specified and no other races',
            6: 'Asian alone',
            7: 'Native Hawaiian and Other Pacific Islander alone',
            8: 'Some Other Race alone',
            9: 'Two or More Races',
            float('nan'): 'N/A'},
    'RAC1PR': {1: 'White alone',
            2: 'Black or African American alone',
            3: 'Other',
            4: 'Other',
            5: 'Other',
            6: 'Asian alone',
            7: 'Other',
            8: 'Other',
            9: 'Other',
            float('nan'): 'Other'},
}