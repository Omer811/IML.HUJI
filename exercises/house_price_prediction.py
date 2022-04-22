import pandas

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import time
import datetime
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import os
from sklearn.cluster import KMeans


PLOT_FILE_FORMAT = ".png"

MIN_LIVING_SQFT = 20
MIN_LIVING_LOT = 100
MIN_FLOORS = 1
pio.templates.default = "simple_white"
NUM_OF_CLUSTERS = 18



def __cluster_location(df: pd.DataFrame):
    kmeans = KMeans(n_clusters=NUM_OF_CLUSTERS, init='k-means++')
    lat_long = pd.DataFrame()
    lat_long["lat"] = df["lat"]
    lat_long["long"] = df["long"]
    kmeans.fit(lat_long)  # Compute k-means clustering.
    df['cluster_label'] = kmeans.fit_predict(lat_long)
    df.head(10)
    return df


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = df.drop_duplicates()
    df = df.dropna()
    df = df.drop(df[(df["price"] < 0) | (df["bedrooms"] < 0) | (df[
              "sqft_living"] < MIN_LIVING_SQFT) | (df["sqft_lot"] < MIN_LIVING_LOT) |
                    (df["floors"] < MIN_FLOORS)].index)
    df["is_renovated"] = df["yr_renovated"].apply(lambda x: 1 if x > 0 else 0)

    if len(df["zipcode"].unique()) > 100:  # decide what describes the location
        # better
        __cluster_location(df)
        df = df.join(pandas.get_dummies(df["cluster_label"], prefix="cluster"))
        df = df.drop(columns=["cluster_label"])
    else:
        df = df.join(pandas.get_dummies(df["zipcode"], prefix="zipcode"))

    price = df["price"]
    df = df.drop(columns=["price", "id", "date", "zipcode", "long", "lat"])


    return df, price


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    pear_corr = pd.DataFrame(data=np.empty((1, X.shape[1]), dtype=float),
                             columns=X.columns, index=["pearson_corr"])
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for column in pear_corr:
        pear_corr[column] = __calculate_pearson_correlation(X[column], y)[0, 1]
        fig = px.scatter(y=y.to_numpy(dtype=float, na_value=0),
                         x=X[column].to_numpy(dtype=float, na_value=0),
                         labels={'x': column, 'y': "price"},
                         title="Price ""against " + column + " \nPearson "
                                                             "correlation: " + str(
                             pear_corr[column]))
        fig.write_image(output_path + "/scatter_" + column + PLOT_FILE_FORMAT)
    print("min: " + pear_corr.idxmin(axis=1) + " max: " + pear_corr.idxmax(
        axis=1))#todo: delete before submission





def __calculate_pearson_correlation(feature: pd.Series,
                                    response: pd.Series) -> np.ndarray:
    """
    data: (n_samples, n_features) data to correlate
    return: calculated pearson correlation
    """
    feature = feature.to_numpy(dtype=float, na_value=0)
    response = response.to_numpy(dtype=float, na_value=0)
    cov = np.cov(feature.reshape(-1), response.reshape(-1))
    x_std = feature.std()
    y_std = response.std()
    xv, yv = np.meshgrid(x_std, y_std)
    return cov / (xv * yv)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    data, prices = load_data("datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # pearson = __calculate_pearson_correlation(data["date"],prices)
    feature_evaluation(data, prices, "out")

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(data, prices, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    reg = LinearRegression(include_intercept=False)
    percent = np.arange(0.10, 1.01, 0.01)
    loss = []
    for p in percent:
        sample_loss = []
        for i in range(10):
            s_x = train_x.sample(frac=p)
            s_y = train_y[s_x.index]
            reg.fit(s_x.to_numpy(dtype=float, na_value=0),
                    s_y.to_numpy(dtype=float,
                                 na_value=0))
            sample_loss.append(
                reg.loss(test_x.to_numpy(dtype=float, na_value=0),
                         test_y.to_numpy(dtype=float, na_value=0)))
        loss.append(sample_loss)
    loss = np.array(loss)
    mean_loss = loss.mean(axis=1)
    error = 2 * loss.std(axis=1)
    y_upper = mean_loss + error
    y_lower = mean_loss - error

    x = percent
    fig = go.Figure([
        go.Scatter(
            name="Test",
            x=x,
            y=mean_loss,
            line=dict(color='rgb(0,100,80)'),
            mode='lines',
        ),
        go.Scatter(
            x=x,
            y=y_upper,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ), go.Scatter(
            x=x,
            y=y_lower,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    fig.update_layout(yaxis_title="Loss", xaxis_title="Sample Size",
                      title="loss based on sample size")
    fig.show()
