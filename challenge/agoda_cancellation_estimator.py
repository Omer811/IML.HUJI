from __future__ import annotations
from typing import NoReturn

import pandas as pd

from IMLearn.base import BaseEstimator
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,HistGradientBoostingClassifier
from IMLearn.metrics import mean_square_error
from datetime import time

class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        super().__init__()
        self.g = GradientBoostingClassifier()
        self.a = AdaBoostClassifier(algorithm="SAMME")

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """

        # self.h = HistGradientBoostingClassifier(categorical_features=)

        # self.g.fit(X,y)
        self.a.fit(X,y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        pred =  list(self.a.predict(X))
        print(sorted(pred))
        for i in range(len(pred)):
            pred[i] = self.is_in_bound(pred[i])

        return np.array(pred)


    def is_in_bound(self, date):
        return date>=np.datetime64("2018-12-07")\
               and  date<=np.datetime64("2018-12-13")

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """
        return mean_square_error(self.a.predict(X),y)
