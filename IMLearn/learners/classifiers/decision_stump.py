from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from IMLearn.metrics.loss_functions import misclassification_error
from itertools import product




class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        min_loss = 1
        for i in range(X.shape[1]):
            for sign in [-1,1]:
                thres,loss = self._find_threshold(X[:,i],y,sign)
                if loss<min_loss:
                    min_loss=loss
                    self.threshold_,self.j_,self.sign_=thres,i,sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        predict = np.empty(X.shape[0])
        predict[X[:, self.j_] < self.threshold_] = -self.sign_
        predict[X[:, self.j_] >= self.threshold_] = self.sign_
        return predict


    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        indexing = np.argsort(values)
        labels = labels[indexing]
        values = values[indexing].astype(float)

        threshold_placement = np.concatenate([[-np.inf],(values[1:] + values[
                                             :-1])/ 2,[np.inf]])
        first_threshold_gain = np.sum(labels[np.sign(labels)==sign])
        all_threshold_gains = np.append(first_threshold_gain,
                                             first_threshold_gain-np.cumsum(
                                                 labels*sign))
        best_gain = np.argmax(all_threshold_gains)
        chosen_labels = np.empty(labels.size)
        chosen_labels[values<threshold_placement[best_gain]] = -sign
        chosen_labels[values>=threshold_placement[best_gain]] = sign

        miss_loss = np.sum(np.abs(labels[np.sign(chosen_labels) != np.sign(
            labels)]), axis=0) / np.sum(np.abs(labels))
        return threshold_placement[best_gain],miss_loss


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.predict(X)
        return np.sum(np.abs(y[np.sign(y)!=y_pred]))
