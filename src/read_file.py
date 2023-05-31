# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Original code from this repo on GitHub

minimax-fair
    Repo URL https://github.com/amazon-science/minimax-fair
    Written by gillwesl-aws and jimmyren23 (Jimmy Ren)
    Distributed under Apache 2.0 license
    
No modification.
"""

import numpy as np


def read_dataset_from_file(file_path, ):
    data = np.load(file_path, allow_pickle=True)
    X = data['X']
    Y = data['Y']
    grouplabels = data['grouplabels']
    try:
        group_sets = data['group_sets']
    except KeyError:
        group_sets = data['group_set']
    group_types = data.get('group_types', [])
    is_binary = data['is_binary']
    return X, Y, grouplabels, group_sets, group_types, is_binary

