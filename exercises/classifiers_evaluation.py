from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd


pio.templates.default = "simple_white"
X, y = None, None
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    global X, y
    data = np.load(filename)
    X = data[:, 0:2]
    y = data[:, 2]
    return X, y


def callback_fun(fit: Perceptron, cur_sample: np.ndarray, cur_response: int):
    global X, y
    fit.training_loss.append(fit.loss(X, y))


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    global X, y
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration

        perc = Perceptron(include_intercept=True, callback=callback_fun)
        perc.fit(X, y)
        # Plot figure
        df = pd.DataFrame({"Iteration": [i for i in range(len(perc.training_loss))],
                           "Loss": perc.training_loss})
        fig = px.line(df, x="Iteration", y="Loss",title="Loss per Iterations "
                                                        "for "+n)
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X,y = load_dataset(f)


        # Fit models and predict over training set
        l = LDA()
        g = GaussianNaiveBayes()
        l.fit(X,y)
        g.fit(X,y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        l_y_pred = l.predict(X)
        g_y_pred = g.predict(X)
        l_accuracy = accuracy(y,l_y_pred)
        g_accuracy = accuracy(y,g_y_pred)
        df = pd.DataFrame({"sample_x":X[:,0],"sample_y":X[:,1],"true_y":y,
                           "predicted_y_LDA":l_y_pred,"predicted_y_GNB":g_y_pred})
        fig = make_subplots(1,2,subplot_titles=("LDA accuracy: "
                                                ""+str(l_accuracy),
                                                "GNB accuracy: "
                                                ""+str(g_accuracy)))
        fig.add_trace(go.Scatter(x=df["sample_x"],y=df["sample_y"],
                                 marker_color =df["predicted_y_LDA"],
                                 marker_symbol = df["true_y"],
                                 mode="markers",marker=dict(size=10)),
                      row=1,
                      col=1)
        fig.add_trace(go.Scatter(x=g.mu_[:,0], y=g.mu_[:,1],
                                 mode="markers", marker=dict(size=30,
                                 color="black",symbol="x")),row=1, col=1)
        fig.add_trace(go.Scatter(x=df["sample_x"], y=df["sample_y"],
                                 marker_color=df["predicted_y_GNB"],
                                 marker_symbol=df["true_y"],
                                 mode="markers", marker=dict(size=10)),
                                 row=1, col=2)
        fig.add_trace(go.Scatter(x=l.mu_[:, 0], y=l.mu_[:, 1],
                                 mode="markers", marker=dict(size=30,
                                 color="black",symbol="x")),
                                 row=1, col=2)
        for i in range(l.classes_.size):
            # e_val,e_vec = np.linalg.eig(l.cov_)
            # e_x,e_y = ellipse(l.mu_[i][0],l.mu_[i][1],a=np.sqrt(e_val[0]),
            #                   b=np.sqrt(e_val[1]),ax1 = e_vec[:,0],ax2=e_vec[:,1])
            fig.add_trace(get_ellipse(l.mu_[i],l.cov_),row=1, col=1)
        for i in range(g.classes_.size):
            # e_val,e_vec = np.linalg.eig(g.vars_[i])
            # e_x,e_y = ellipse(g.mu_[i][0],g.mu_[i][1],a=np.sqrt(e_val[0]),
            #                   b=np.sqrt(e_val[1]),ax1 = e_vec[:,0], ax2=e_vec[:,1])
            fig.add_trace(get_ellipse(g.mu_[i],g.vars_[i]),row=1, col=2)
        fig.update_layout(title_text="LDA and GNB prediction, dataset: "
                                     ""+str(f),
                          showlegend=False)
        fig.show()


# def ellipse(x_center=0, y_center=0, ax1=[1, 0], ax2=[0, 1], a=1, b=1, N=100):
#     # x_center, y_center the coordinates of ellipse center
#     # ax1 ax2 two orthonormal vectors representing the ellipse axis directions
#     # a, b the ellipse parameters
#     if not ((np.isclose(np.linalg.norm(ax1),1) and np.isclose(np.linalg.norm(
#             ax2),1))):
#         raise ValueError('ax1, ax2 must be unit vectors')
#     if abs(np.dot(ax1, ax2)) > 1e-06:
#         raise ValueError('ax1, ax2 must be orthogonal vectors')
#     # rotation matrix
#     R = np.array([ax1, ax2]).T
#     if np.linalg.det(R) < 0:
#         raise ValueError(
#             "the det(R) must be positive to get a  positively oriented ellipse reference frame")
#     t = np.linspace(0, 2 * np.pi, N)
#     # ellipse parameterization with respect to a system of axes of directions a1, a2
#     xs = 2*a * np.cos(t)
#     ys = 2*b * np.sin(t)
#
#     # coordinate of the  ellipse points with respect to the system of axes [1, 0], [0,1] with origin (0,0)
#     xp, yp = np.dot(R, [xs, ys])
#     x = xp + x_center
#     y = yp + y_center
#     return x, y
if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
