from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.random.uniform(-1.2, 2, n_samples)
    sorted_X = np.sort(X)
    noise_per_point = np.random.normal(0, noise, n_samples)
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)

    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(X),
                            pd.Series(f(X) + noise_per_point),2 / 3)
    train_x, train_y, test_x, test_y = train_x.to_numpy().reshape(-1), \
                    train_y.to_numpy(), test_x.to_numpy().reshape(-1),\
                                       test_y.to_numpy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sorted_X, y=f(sorted_X), name="model",
                             mode="lines",
                             showlegend=True,
                             marker=dict(color="black")))
    fig.add_trace(go.Scatter(x=test_x, y=test_y, name="test",
                             mode="markers",
                             showlegend=True,
                             marker=dict(color="blue")))
    fig.add_trace(go.Scatter(x=train_x, y=train_y, name="train",
                             mode="markers",
                             showlegend=True,
                             marker=dict(color="red")))
    fig.update_layout(title_text = "Data divided into train and test")
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_errors = []
    validation_errors = []

    ks = np.arange(11)
    for k in ks:
        validation, train = cross_validate(PolynomialFitting(k), train_x,
                                           train_y, mean_square_error, 5)
        validation_errors.append(validation)
        train_errors.append(train)
    train_errors, validation_errors = np.array(train_errors), np.array(
        validation_errors)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=train_errors, name="avg_train_error",
                             mode="lines+markers",
                             showlegend=True,
                             marker=dict(color="red")))
    fig.add_trace(go.Scatter(x=ks, y=validation_errors,
                             name="avg_validation_error",
                             mode="lines+markers",
                             showlegend=True,
                             marker=dict(color="blue")))
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = ks[np.argmin(validation_errors)]
    polyfit = PolynomialFitting(k_star)
    polyfit.fit(train_x,train_y)
    print(f"Error on test set with k={k_star} is:"
          f" {np.round(polyfit.loss(test_x,test_y),2)}")



def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X,y = datasets.load_diabetes(return_X_y=True)
    # X, y = datasets.load_iris(return_X_y=True)
    X_train,y_train = X[:n_samples],y[:n_samples]
    X_test,y_test = X[n_samples:],y[n_samples:]


    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    eps = 1 / 1000
    lam = np.linspace(0 +eps, 4, n_evaluations)
    models = [RidgeRegression, Lasso]
    models_name = ["Ridge", "Lasso"]
    model_count = len(models)
    models_train_error = np.empty((model_count, n_evaluations))
    models_validation_error = np.empty((model_count, n_evaluations))
    #models_coef = np.empty((model_count,n_evaluations,X.shape[1]))
    for model in range(model_count):
        for i in range(n_evaluations):

            m = models[model](lam[i])
            validation_error, train_error = cross_validate(
                    m, X_train, y_train, mean_square_error,5)
            models_train_error[model, i] = train_error
            models_validation_error[model, i] = validation_error
            # if models_name[model] == "Ridge":
            #     models_coef[model,i] = m.coefs_
            # else:
            #     models_coef[model, i] = m.coef_
    # fig = go.Figure()
    # for model in range(model_count):
    #     for coef in range(1,2):
    #      fig.add_trace(go.Scatter(x=lam, y=models_coef[model,:,coef],
    #                              name=f"coef {coef} on {models_name[model]}",
    #                              mode="lines+markers",
    #                              showlegend=True))
    # fig.show()
    fig = go.Figure()
    for model in range(model_count):
        fig.add_trace(go.Scatter(x=lam, y=models_train_error[model],
                                 name=f"avg_train_error on {models_name[model]}",
                                 mode="lines",
                                 showlegend=True))
        fig.add_trace(go.Scatter(x=lam, y=models_validation_error[model],
                                 name=f"avg_validation_error on "
                                      f"{models_name[model]}",
                                 mode="lines",
                                 showlegend=True))
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    min_val_err = np.argmin(models_validation_error,axis=1)
    min_lams = [lam[min_val_err[model]] for model in range (model_count)]
    for model in range(model_count):
        print(f"Minimal validation error for {models_name[model]} achieved "
              f"using lambda = {min_lams[model]}")
    models.append(LinearRegression)
    models_name.append("Linear Regression")
    model_count = len(models)
    model_test_error = np.empty((model_count,len(min_lams)))
    for model in range(model_count):
        for min_lam in range(len(min_lams)):
            if models_name[model] == "Linear Regression":
                m = models[model](False)
            else:
                m = models[model](min_lams[min_lam])
            m.fit(X_train,y_train)
            if models_name[model] == "Lasso":
                model_test_error[model, min_lam] = mean_square_error(
                    y_test, m.predict(X_test))
            else:
                model_test_error[model, min_lam] = m.loss(X_test, y_test)
            print(f"Test error on {models_name[model]} with lam = "
                  f"{0 if models_name[model] == 'Linear Regression' else min_lams[min_lam]} was "
                  f"{model_test_error[model, min_lam]}")




if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    # select_polynomial_degree(100,noise=5)
    # select_polynomial_degree(100,noise=0)
    # select_polynomial_degree(1500,noise=10)
    select_regularization_parameter()
