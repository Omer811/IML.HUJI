from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from ...metrics.loss_functions import misclassification_error

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.pi_ = np.zeros_like(self.classes_)
        self.mu_ = np.zeros((self.classes_.size, X.shape[1]))
        cov = []
        for i in range(self.classes_.size):
            diff = y == self.classes_[i]
            nk = np.count_nonzero(diff)
            self.pi_[i] = float(nk / y.size)
            self.mu_[i] = (X[diff]).mean(axis=0)

            x_mu = X[diff]-self.mu_[i]
            # cov.append(np.matmul(x_mu.T, x_mu, axes=[(0, 1), (0, 1), (-2,
            #                     -1)]) / (X[diff].shape[0]))
            cov.append(np.diag(np.power(x_mu,2).mean(axis=0)))
        self.vars_ = np.array(cov)
        print("hi")

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
        return self.classes_[np.argmax(self.likelihood(X),axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        classification = np.zeros((X.shape[0], self.classes_.size))
        for i in range(self.classes_.size):
            cov_inv = np.linalg.pinv(self.vars_[i])
            a = np.matmul(cov_inv, self.mu_.T,
                          axes=[(0, 1), (0, 1), (-1, -2)])
            b = np.log(self.pi_) - 0.5 * np.diag(
                np.matmul(self.mu_, a.T, axes=[(0, 1), (0, 1), (-2, -1)]))

            classification[:, i] = a[i] @ X.T + b[i]
        return classification

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
        return misclassification_error(y,self.predict(X))
