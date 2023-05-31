# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Original code from this repo on GitHub

minimax-fair
Repo URL https://github.com/amazon-science/minimax-fair
Written by gillwesl-aws (https://github.com/gillwesl) and jimmyren23 (Jimmy Ren, https://github.com/jimmyren23)
Distributed under Apache 2.0 license

Modified to add GPU support for training MLP models
"""

from abc import ABC

import torch
from torch import nn
import scipy.special
import numpy as np
from sklearn.metrics import accuracy_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TorchMLP(nn.Module, ABC):
    """
    A feedforward NN in pytorch using ReLU activiation functions between all layers but the last
    which uses a sigmoid activiation function. Supports an arbitrary number of hidden layers.
    """

    def __init__(self, h_sizes, out_size=1, task='classification'):
        """
        :param h_sizes: input sizes for each hidden layer (including the first)
        :param out_size: defaults to 1 for binary and represents the (positive class probability?)
        :param task: 'classification' or 'regression'
        """
        super(TorchMLP, self).__init__()

        # Hidden layers
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))

        # Output layer
        self.out = nn.Linear(h_sizes[-1], out_size)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # Feedforward
        for layer in self.hidden:
            x = self.relu(layer(x))
        output = self.out(x)  # Sigmoid applied later in BCEWithLogitLoss, and applied automatically in predict_proba
        return output.double()


class MLPClassifier:
    """
    Wrapper class so our MLP looks like an sklearn model
    """

    def __init__(self, h_sizes, lr=0.0001, momentum=0.9, weight_decay=0, task='classification'):
        self.model = TorchMLP(h_sizes)
        self.model.double()  # set model type to double
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    def fit(self, X, y, sampleweights, n_epochs, loss_type='BCE'):
        """
        Fits the model using the entire sample data as the batch size
        """
        X = torch.from_numpy(X)
        y = torch.from_numpy(y).double()
        self.model.train()  # Puts model in training mode so it updates itself

        # Binary Cross-Entropy Loss with sample weights
        criterion = nn.BCEWithLogitsLoss(weight=torch.from_numpy(sampleweights))  # convert weights to tensor

        for epoch in range(n_epochs):
            self.optimizer.zero_grad()  # Set gradients to 0 before back propagation for this epoch
            # Forward pass
            y_pred = self.model(X)
            # Compute Loss
            loss = criterion(y_pred.squeeze(), y)
            # print(f'Epoch {epoch}: train loss: {loss.item()}')
            # Backward pass
            loss.backward()
            self.optimizer.step()

        return self

    def predict_proba(self, X):
        """
        :param X: Feature matrix we want to make predictions on
        :return: Column vector of prediction probabilities, one for each row (instance) in X
        """
        self.model.eval()  # Puts the model in evaluation mode so calls to forward do not update it
        with torch.no_grad():  # Disables automatic gradient updates from pytorch since we are just evaluating
            return torch.sigmoid(self.model(torch.from_numpy(X))).numpy().squeeze()  # Apply sigmoid manually

    def predict(self, X):
        """
        :param X: Feature matrix we want to make predictions on
        :return: Binary predictions for each instance of X
        """
        return self.predict_proba(X) > 0.5  # Converts probabilistic predictions into binary ones

class MLPClassifierGPU:
    """
    Wrapper class so our MLP looks like an sklearn model
    """

    def __init__(self, h_sizes, lr=0.0001, momentum=0.9, weight_decay=0, task='classification'):
        self.model = TorchMLP(h_sizes)
        self.model.double()  # set model type to double
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def to(self, device):
        self.model.to(device)

    @torch.no_grad()
    def evaluate(self, X, y):
        self.model.eval()
        prediction = self.model(X).cpu().numpy()
        target = y.cpu().numpy()
        prediction = np.round(scipy.special.expit(prediction))
        score = accuracy_score(target, prediction)
        return score
    
    def fit(self, X, y, sampleweights, n_epochs, loss_type='BCE'):
        """
        Fits the model using the entire sample data as the batch size
        """
        X = torch.from_numpy(X)
        y = torch.from_numpy(y).double()
        X, y = X.to(device), y.to(device)
        self.model.to(device)

        # Binary Cross-Entropy Loss with sample weights
        sampleweights = torch.from_numpy(sampleweights)
        sampleweights = sampleweights.to(device)
        criterion = nn.BCEWithLogitsLoss(weight=sampleweights)  # convert weights to tensor

        report_frequency = 1000
        
        for epoch in range(n_epochs):
            self.model.train()  # Puts model in training mode so it updates itself
            self.optimizer.zero_grad()  # Set gradients to 0 before back propagation for this epoch
            # Forward pass
            y_pred = self.model(X)  # outputs are logits
            # Compute Loss
            loss = criterion(y_pred.squeeze(), y)
            # Backward pass
            loss.backward()
            self.optimizer.step()
            if epoch%report_frequency==0:
                loss_value = loss.item()
                score = self.evaluate(X, y)
                print(f'Epoch {epoch}: train accuracy: {score} | train loss: {loss_value}')
            
        self.model.to(torch.device('cpu'))

        return self

    def predict_proba(self, X):
        """
        :param X: Feature matrix we want to make predictions on
        :return: Column vector of prediction probabilities, one for each row (instance) in X
        """
        self.model.eval()  # Puts the model in evaluation mode so calls to forward do not update it
        with torch.no_grad():  # Disables automatic gradient updates from pytorch since we are just evaluating
            return torch.sigmoid(self.model(torch.from_numpy(X))).numpy().squeeze()  # Apply sigmoid manually

    def predict(self, X):
        """
        :param X: Feature matrix we want to make predictions on
        :return: Binary predictions for each instance of X
        """
        return self.predict_proba(X) > 0.5  # Converts probabilistic predictions into binary ones
