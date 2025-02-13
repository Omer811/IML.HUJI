from __future__ import annotations
from typing import Callable, NoReturn
import numpy as np

from IMLearn.base import BaseModule, BaseLR
from .learning_rate import FixedLR

OUTPUT_VECTOR_TYPE = ["last", "best", "average"]


def default_callback(**kwargs) -> NoReturn:
    pass


class GradientDescent:
    """
    Gradient Descent algorithm

    Attributes:
    -----------
    learning_rate_: BaseLR
        Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

    tol_: float
        The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
        specified tolerance

    max_iter_: int
        The maximum number of GD iterations to be performed before stopping training

    out_type_: str
        Type of returned solution:
            - `last`: returns the point reached at the last GD iteration
            - `best`: returns the point achieving the lowest objective
            - `average`: returns the average point over the GD iterations

    callback_: Callable[[...], None], default=default_callback
        A callable function to be called after each update of the model while fitting to given data.
        Callable function receives as input any argument relevant for the current GD iteration. Arguments
        are specified in the `GradientDescent.fit` function
    """

    def __init__(self,
                 learning_rate: BaseLR = FixedLR(1e-3),
                 tol: float = 1e-5,
                 max_iter: int = 1000,
                 out_type: str = "last",
                 callback: Callable[
                     [GradientDescent, ...], None] = default_callback):
        """
        Instantiate a new instance of the GradientDescent class

        Parameters
        ----------
        learning_rate: BaseLR, default=FixedLR(1e-3)
            Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

        tol: float, default=1e-5
            The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
            specified tolerance

        max_iter: int, default=1000
            The maximum number of GD iterations to be performed before stopping training

        out_type: str, default="last"
            Type of returned solution. Supported types are specified in class attributes

        callback: Callable[[...], None], default=default_callback
            A callable function to be called after each update of the model while fitting to given data.
            Callable function receives as input any argument relevant for the current GD iteration. Arguments
            are specified in the `GradientDescent.fit` function
        """
        self.learning_rate_ = learning_rate
        if out_type not in OUTPUT_VECTOR_TYPE:
            raise ValueError("output_type not supported")
        self.out_type_ = out_type
        self.tol_ = tol
        self.max_iter_ = max_iter
        self.callback_ = callback

    def fit(self, f: BaseModule, X: np.ndarray, y: np.ndarray):
        """
        Optimize module using Gradient Descent iterations over given input samples and responses

        Parameters
        ----------
        f : BaseModule
            Module of objective to optimize using GD iterations
        X : ndarray of shape (n_samples, n_features)
            Input data to optimize module over
        y : ndarray of shape (n_samples, )
            Responses of input data to optimize module over

        Returns
        -------
        solution: ndarray of shape (n_features)
            Obtained solution for module optimization, according to the specified self.out_type_

        Notes
        -----
        - Optimization is performed as long as self.max_iter_ has not been reached and that
        Euclidean norm of w^(t)-w^(t-1) is more than the specified self.tol_

        - At each iteration the learning rate is specified according to self.learning_rate_.lr_step

        - At the end of each iteration the self.callback_ function is called passing self and the
        following named arguments:
            - solver: GradientDescent
                self, the current instance of GradientDescent
            - weights: ndarray of shape specified by module'cur_score weights
                Current weights of objective
            - val: ndarray of shape specified by module'cur_score compute_output function
                Value of objective function at current point, over given data X, y
            - grad:  ndarray of shape specified by module'cur_score compute_jacobian function
                Module'cur_score jacobian with respect to the weights and at current point, over given data X,y
            - t: int
                Current GD iteration
            - eta: float
                Learning rate used at current iteration
            - delta: float
                Euclidean norm of w^(t)-w^(t-1)

        """
        prev_w = None
        cur_w = f.weights
        best_w = cur_w.copy()
        best_w_score = f.compute_output(X=X,y=y)
        avg_w = cur_w.copy()
        iter_num = 0
        while iter_num < self.max_iter_ and (prev_w is None or
                                             self.compute_delta(prev_w,
                                                                cur_w) > self.tol_):
            prev_w = cur_w.copy()
            cur_w = cur_w + \
                    self.learning_rate_.lr_step(
                        t=iter_num) * -f.compute_jacobian(X=X,y=y)
            f.weights = cur_w
            cur_score = f.compute_output(X=X,y=y)
            avg_w, best_w, best_w_score = self.update_weights(avg_w, best_w,
                                                              best_w_score,
                                                              cur_w, cur_score)
            self.callback_(solver=self, weights=cur_w, val=cur_score,
                           grad=f.compute_jacobian(X=X,y=y), t=iter_num,
                           eta=self.learning_rate_.lr_step(t=iter_num),
                           delta=self.compute_delta(prev_w, cur_w))
            iter_num += 1
        if self.out_type_ == 'best':
            return best_w
        if self.out_type_ == "average":
            return avg_w / iter_num
        if self.out_type_ == 'last':
            return f.weights

    def update_weights(self, avg_w, best_w, best_w_score, cur_w, s):
        updated_score = best_w_score
        if self.out_type_ == 'best':
            if s < best_w_score:
                updated_score = s
                best_w = cur_w.copy()
        if self.out_type_ == "average":
            avg_w += cur_w
        return avg_w, best_w, updated_score

    def compute_delta(self, prev_w, cur_w):
        if prev_w is None: return True
        return np.abs(np.linalg.norm(cur_w-prev_w, ord=2))
