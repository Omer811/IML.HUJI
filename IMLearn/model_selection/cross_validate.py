from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator

def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    x_ind = np.arange(X.shape[0])
    split = np.array_split(x_ind,cv)
    validate_error = []
    train_error = []

    for i in range(cv):
        cur_data_ind = np.concatenate(split[:i] + split[i+1:], axis=0)
        cur_train_X = X[cur_data_ind]
        cur_train_y = y[cur_data_ind].reshape(-1,1)
        estimator.fit(cur_train_X,cur_train_y)
        y_pred_fold = estimator.predict(X[split[i]])
        validate_error.append(scoring(y_pred_fold,y[split[i]]))
        y_pred_train = estimator.predict(cur_train_X)
        train_error.append(scoring(cur_train_y.reshape(-1),y_pred_train))

    validate_error =np.array(validate_error)
    train_error = np.array(train_error)
    return validate_error.mean() ,train_error.mean()