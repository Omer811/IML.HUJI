import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule, BaseLR
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.metrics import misclassification_error
from sklearn.metrics import roc_curve
import plotly.graph_objects as go
from IMLearn.model_selection import cross_validate


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    raise NotImplementedError()


def run_gd(model_type:Type[BaseModule],title:str,f: BaseModule,
           learning_rate: BaseLR = FixedLR(1e-3),
                 tol: float = 1e-5,
                 max_iter: int = 1000,
                 out_type: str = "last")->(go.Figure,go.Figure,int):
    losses_ = np.empty(max_iter+1)
    weights_ = np.empty((max_iter+1, f.shape[0]))
    losses_[0] = f.compute_output()
    weights_[0] = f.weights
    cur_t = [0]
    def callback_func(solver, weights, val, grad, t, eta, delta):
        losses_[t+1] = val
        weights_[t+1] = weights
        cur_t[0] = t
    gd = GradientDescent(learning_rate=learning_rate,tol=tol,
                         max_iter=max_iter,out_type=out_type,
                         callback=callback_func)
    gd.fit(f,None,None)
    losses_ = losses_[:cur_t[0]+2]
    weights_ = weights_[:cur_t[0]+2]
    min_loss = np.round(np.min(losses_),2)
    descent = plot_descent_path(model_type,weights_,title=title+f" iteration: "
                                        f"{cur_t[0]} loss: {min_loss}")
    conv = go.Scatter(x=np.arange(losses_.size),y=losses_)

    return descent,conv,min_loss


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01,
                                                       .001)):
    models = [L1,L2]
    models_text = ["L1","L2"]
    losses = go.Figure()
    losses.layout.title = "convergence"
    for i in range(len(models)):
        for eta in etas:
            f = models[i](init)
            descent,conv,min_loss = run_gd(model_type=models[i],
                                  learning_rate=FixedLR(eta),
                          title=f"Plot for {models_text[i]} with eta ="
                                  f" {eta}",f=f)
            descent.write_image(f"graphs/{models_text[i]}_eta_{eta}.png",
                                width=1500,height=1000)
            conv.name= f"{models_text[i]} with eta = {eta}"
            losses.add_trace(conv)

            print(f"{models_text[i]} with eta = {eta}, min loss = {min_loss}")
    losses.write_image("graphs/fixed_conv.png",width=1500,height=1000)



def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    models = [L1,L2]
    models_text = ["L1", "L2"]
    losses = go.Figure()
    losses.layout.title = "convergence"
    for i in range(len(models)):
        for gamma in gammas:
            f = models[i](init)
            descent, conv, min_loss = run_gd(model_type=models[i],
                          learning_rate=ExponentialLR(eta,gamma),
                          title=f"Plot for {models_text[i]} with eta ="
                          f" {eta} and gamma = {gamma}", f=f)
            descent.write_image(f"graphs/{models_text[i]}_eta_"
                                f"{eta}_gamma_{gamma}.png",
                                width=1500, height=1000)
            conv.name = f"{models_text[i]} with eta = {eta} gamma = {gamma}"
            losses.add_trace(conv)

            print(f"{models_text[i]} with eta = {eta},gamma ={gamma} min loss ="
                  f" {min_loss}")
    losses.write_image("graphs/exp_conv.png", width=1500, height=1000)





def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    solver = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000)
    log_reg = LogisticRegression(solver=solver)
    alphas = np.arange(0,1.01,step=0.01)
    log_reg.fit(X_train.to_numpy(),y_train.to_numpy())
    probs = log_reg.predict_proba(
        X_train.to_numpy())
    fpr, tpr, thrs = roc_curve(y_train.to_numpy(),probs)
    fig = go.Figure()
    fig.layout.title = f"ROC Curve"
    fig.add_trace(go.Scatter(x=fpr,y=tpr))
    fig.write_image("graphs/roc_curve.png", width=1500, height=1000)

    tpr_a = np.empty_like(alphas)
    fpr_a = np.empty_like(alphas)
    for i in range(alphas.size):
        y_pred = np.where(probs >= alphas[i], 1, 0)
        cur_fpr, cur_tpr, cur_thresholds = roc_curve(y_train.to_numpy(), y_pred)
        tpr_a[i] = cur_tpr[1]
        fpr_a[i] = cur_fpr[1]

    best_alpha = alphas[np.argmax(tpr_a-fpr_a)]
    print(f"best alpha:{np.round(best_alpha,2)}")
    log_reg.alpha_ = best_alpha
    best_alpha_loss = misclassification_error(y_test.to_numpy(),log_reg.predict(
        X_test.to_numpy()))
    print(f"loss on test for best alpha:{np.round(best_alpha_loss, 2)}")
    penalties = ["l1","l2"]
    lams = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    for penalty in penalties:
        train_errors = np.empty_like(lams)
        validation_errors = np.empty_like(lams)
        for i in range(lams.size):
            log_reg = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000), alpha=0.5,
                                         penalty=penalty, lam=lams[i])
            validation, train = cross_validate(log_reg,X_train.to_numpy(),
                             y_train.to_numpy(), misclassification_error, 5)
            validation_errors[i] = validation
            train_errors[i] = train
        best_lam = lams[int(np.argmin(validation_errors))]
        print(f"for penalty {penalty} best lam:{best_lam}")
        log_reg = LogisticRegression(penalty=penalty,lam=best_lam)
        log_reg.fit(X_train.to_numpy(),y_train.to_numpy())
        y_pred = log_reg.predict(X_test.to_numpy())
        error = misclassification_error(y_true=y_test.to_numpy(),y_pred=y_pred)
        print(f"for penalty {penalty} misclassification_error:"
              f"{np.round(error,2)}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
