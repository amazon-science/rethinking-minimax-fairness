"""Implements abstract classes for folktables data source and problem definitions."""

"""
2023-05-31: Amazon modification

We made a slight modification to the original version of the code in the 
folktables package: the df_to_pandas function is modified to include an 
argument 'drop_first' which drops the first level of categorical variables
after making the dummies

Repo URL (released under MIT License)
    https://github.com/socialfoundations/folktables
File URL
    https://github.com/socialfoundations/folktables/blob/main/folktables/folktables.py
Authors
    https://github.com/jenno-verdonck
    https://github.com/millerjohnp
    https://github.com/mrtzh
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class DataSource(ABC):
    """Provides access to data source."""

    @abstractmethod
    def get_data(self, **kwargs):
        """Get data sample from universe.

        Returns:
            Sample."""
        pass


class Problem(ABC):
    """Abstract class for specifying learning problem."""

    @abstractmethod
    def df_to_numpy(self, df):
        """Return learning problem as numpy array."""
        pass

    # Returns the column name
    @property    
    @abstractmethod
    def target(self):
        pass

    @property
    @abstractmethod
    def features(self):
        pass

    @property
    @abstractmethod
    def target_transform(self):
        pass


class BasicProblem(Problem):
    """Basic prediction or regression problem."""

    def __init__(self,
                 features,
                 target,
                 target_transform=None,
                 group=None,
                 group_transform=lambda x: x,
                 preprocess=lambda x: x,
                 postprocess=lambda x: x):
        """Initialize BasicProblem.

        Args:
            features: list of column names to use as features
            target: column name of target variable
            target_transform: feature transformation for target variable
            group: designated group membership feature
            group_transform: feature transform for group membership
            preprocess: function applied to initial data frame
            postprocess: function applied to final numpy data array
        """
        self._features = features
        self._target = target
        self._target_transform = target_transform
        self._group = group
        self._group_transform = group_transform
        self._preprocess = preprocess
        self._postprocess = postprocess

    def df_to_numpy(self, df):
        """Return data frame as numpy array.
        
        Args:
            DataFrame.
        
        Returns:
            Numpy array, numpy array, numpy array"""

        df = self._preprocess(df)
        res = []
        for feature in self.features:
            res.append(df[feature].to_numpy())
        res_array = np.column_stack(res)
        
        if self.target_transform is None:
            target = df[self.target].to_numpy()
        else:
            target = self.target_transform(df[self.target]).to_numpy()
        
        if self._group:
            group = self.group_transform(df[self.group]).to_numpy()
        else:
            group = np.zeros(len(target))

        return self._postprocess(res_array), target, group

    def df_to_pandas(self, df, categories=None, dummies=False, drop_first=True):
        """Filters and processes a DataFrame (received from ```ACSDataSource''').
        
        Args:
            df: pd.DataFrame (received from ```ACSDataSource''')
            categories: nested dict with columns of categorical features
                and their corresponding encodings (see examples folder)
            dummies: bool to indicate the creation of dummy variables for
                categorical features (see examples folder)
            --- Amazon modification ---
            drop_first: bool indicating whether to drop the first level of 
                categorical variables after making the dummies
        
        Returns:
            pandas.DataFrame."""
        
        df = self._preprocess(df)

        variables = df[self.features]

        if categories:
            variables = variables.replace(categories)
        
        if dummies:
            variables = pd.get_dummies(variables, drop_first=drop_first)

        variables = pd.DataFrame(self._postprocess(variables.to_numpy()),
                                 columns=variables.columns)

        if self.target_transform is None:
            target = df[self.target]
        else:
            target = self.target_transform(df[self.target])

        target = pd.DataFrame(target).reset_index(drop=True)

        if self._group:
            group = self.group_transform(df[self.group])
            group = pd.DataFrame(group).reset_index(drop=True)
        else:
            group = pd.DataFrame(0, index=np.arange(len(target)), columns=["group"])

        return variables, target, group

    @property
    def target(self):
        return self._target
    
    @property
    def target_transform(self):
        return self._target_transform
    
    @property
    def features(self):
        return self._features
    
    @property
    def group(self):
        return self._group

    @property
    def group_transform(self):
        return self._group_transform
