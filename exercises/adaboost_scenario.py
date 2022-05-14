import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    a_boost = AdaBoost(wl=DecisionStump,iterations=n_learners)
    a_boost.fit(train_X,train_y)
    train_err = np.empty(n_learners)
    test_err = np.empty(n_learners)
    for i in range(n_learners):
        train_err[i] = a_boost.partial_loss(train_X, train_y, i+1)
        test_err[i] = a_boost.partial_loss(test_X, test_y, i+1)
    df = pd.DataFrame()
    df["Train_error"] = train_err
    df["Test_error"] = test_err
    df["Number_of_models"] = np.arange(1,251,1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Number_of_models"],y=df["Train_error"],
                             name = "Train_error" ))
    fig.add_trace(go.Scatter(x=df["Number_of_models"], y=df["Test_error"],
                             name="Test_error"))
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    symbols = np.array(["circle", "x"])
    fig = make_subplots(2,2,subplot_titles=[rf"$\textbf{{{m}}}$" for m in T])
    for i in range(len(T)):
        fig.add_traces([decision_surface(lambda X:a_boost.partial_predict(
            X,T[i]),lims[0],lims[1],showscale=False),
                       go.Scatter(x=test_X[:,0], y=test_X[:,1], mode="markers",
                                  showlegend=False,
                               marker=dict(color=test_y, symbol=symbols[
                                   test_y.astype(int)],
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)) )],
                   rows=(i//2) + 1, cols=(i%2)+1)
    fig.show()
    # Question 3: Decision surface of best performing ensemble
    best_err = 1
    best_size = 0
    for size in range(n_learners):
        cur_err = a_boost.partial_loss(test_X,test_y,size)
        if cur_err<best_err:
            best_err = cur_err
            best_size = size
    fig = go.Figure()
    fig.add_traces([decision_surface(lambda X: a_boost.partial_predict(
        X, best_size), lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                               showlegend=False,
                               marker=dict(color=test_y, symbol=symbols[
                                   test_y.astype(int)],
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black",
                                                     width=1)))])
    fig.update_layout(title=f"Best ensemble size:{best_size}, achieved "
                            f"accuracy:{1-a_boost.partial_loss(test_X,test_y,best_size)}")
    fig.show()
    # Question 4: Decision surface with weighted samples
    normalized_d = (a_boost.D_/np.max(a_boost.D_))*10
    fig = go.Figure()
    fig.add_traces([decision_surface(lambda X: a_boost.partial_predict(
        X, n_learners), lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                               showlegend=False,
                               marker=dict(color=test_y, symbol=symbols[
                                   test_y.astype(int)],size=normalized_d,
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black",
                                                     width=1)))])
    fig.update_layout(title="Full ensemble, marker size correlates to sample "
                            "density")
    fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    #fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)

