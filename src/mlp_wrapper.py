# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Wrapper for the MLP model in the paper
Revisiting Deep Learning Models for Tabular Data
Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko
NeurIPS 2021
"""

from typing import Any, Dict

import numpy as np
import rtdl
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import zero

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TabularMLPClassifier:
    """
    Wrapper class for the MLP architecture benchmarked on tabular data
    From the paper
    Revisiting Deep Learning Models for Tabular Data
    Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko
    NeurIPS 2021
    """

    def __init__(self, d_in, d_layers, d_out=1, lr=0.001, weight_decay=0.0):
        self.model = rtdl.MLP.make_baseline(
            d_in=d_in,
            d_layers=d_layers,
            dropout=0.1,
            d_out=d_out,
        )
        self.model.double()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = F.binary_cross_entropy_with_logits
    
    def apply_model(self, x_num, x_cat=None):
        if isinstance(self.model, rtdl.FTTransformer):
            return self.model(x_num, x_cat)
        elif isinstance(self.model, (rtdl.MLP, rtdl.ResNet)):
            assert x_cat is None
            return self.model(x_num)
        else:
            raise NotImplementedError(
                f'Looks like you are using a custom model: {type(self.model)}.'
                ' Then you have to implement this branch first.'
            )
    
    @torch.no_grad()
    def evaluate(self, X, y):
        self.model.eval()
        prediction = []
        for batch in zero.iter_batches(X, 1024):
            prediction.append(self.apply_model(batch))
        prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
        target = y.cpu().numpy()
        prediction = np.round(scipy.special.expit(prediction))
        score = sklearn.metrics.accuracy_score(target, prediction)
        return score

    def fit(self, X, y, sampleweights, n_epochs):
        """
        Fits the model using the entire sample data as the batch size
        """
        X = torch.from_numpy(X)
        y = torch.from_numpy(y).double()
        X, y = X.to(device), y.to(device)
        self.model.to(device)

        batch_size = 2048
        train_loader = zero.data.IndexLoader(len(X), batch_size, device=device)
        
        # Binary cross-entropy loss with sample weights
        sampleweights = torch.from_numpy(sampleweights)
        sampleweights = sampleweights.to(device)
        

        # report_frequency = len(X) // batch_size

        for epoch in range(1, n_epochs + 1):
            for iteration, batch_idx in enumerate(train_loader):
                self.model.train()
                self.optimizer.zero_grad()
                x_batch = X[batch_idx]
                y_batch = y[batch_idx]
                w_batch = sampleweights[batch_idx]
                loss = self.loss_fn(input=self.apply_model(x_batch).squeeze(1), 
                                    target=y_batch,
                                    weight=w_batch)
                loss.backward()
                self.optimizer.step()
                # if iteration % report_frequency == 0:
                #     print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f}')

            if epoch % 100 == 0:
                loss_value = loss.item()
                train_score = self.evaluate(X, y)
                print(f'Epoch {epoch:03d} | Training score: {train_score:.4f} | Loss: {loss_value:.15f}')

        self.model.to(torch.device('cpu'))

        return self
    
    def predict_proba(self, X):
        """
        :param X: Feature matrix we want to make predictions on
        :return: Column vector of prediction probabilities, one for each row (instance) in X
        """
        self.model.eval()  # Puts the model in evaluation mode so calls to forward do not update it
        with torch.no_grad():  # Disables automatic gradient updates from pytorch since we are just evaluating
            prediction = self.apply_model(torch.from_numpy(X)).squeeze(1).numpy()
            prediction = scipy.special.expit(prediction)  # Apply sigmoid manually
            return prediction

    def predict(self, X):
        """
        :param X: Feature matrix we want to make predictions on
        :return: Binary predictions for each instance of X
        """
        return self.predict_proba(X) > 0.5  # Converts probabilistic predictions into binary ones


