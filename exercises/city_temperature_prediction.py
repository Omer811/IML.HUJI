import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test
from IMLearn.metrics.loss_functions import mean_square_error
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio



MIN_TEMP = -50

pio.templates.default = "simple_white"
import plotly.graph_objects as go


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    df = df.dropna()
    df = df.drop(df[df["Temp"] < MIN_TEMP].index)
    df["Day of Year"] = df["Date"].dt.dayofyear
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_data = data[data["Country"] == "Israel"]
    israel_data = israel_data.astype({"Year": str}, errors="raise")

    fig = px.scatter(israel_data, x="Day of Year", y="Temp", color="Year")
    fig.show()

    months = israel_data.groupby(["Month"])["Temp"].agg([np.std]).reset_index()

    fig = px.bar(months,x="Month",y="std")
    fig.show()

    # Question 3 - Exploring differences between countries

    group_by_country = data.groupby(["Country", "Month"])["Temp"].agg([
        np.mean, np.std]).reset_index()
    fig = px.line(group_by_country, x="Month", y="mean", color='Country',
                  error_y="std", title="Mean Temp by Months per Country")
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    train_x, train_y, test_x, test_y = split_train_test(israel_data["Day of Year"],
                                                        israel_data["Temp"],
                                                        0.75)
    loss = pd.DataFrame()
    for i in range(1, 11):
        poly = PolynomialFitting(i)
        poly.fit(train_x.to_numpy(), train_y.to_numpy(dtype=float))
        loss = loss.append({"Loss": np.round(poly.loss(test_x.to_numpy(),
                          test_y.to_numpy(dtype=float)), decimals=2),
                            "K_Value": i}, ignore_index=True)
    fig = px.bar(loss, x="K_Value", y="Loss")
    fig.show()
    print (loss)

    # Question 5 - Evaluating fitted model on different countries
    min_k = loss["K_Value"][loss["Loss"].idxmin()]
    group_by_country = data.groupby("Country")

    poly = PolynomialFitting(int(min_k))
    poly.fit(group_by_country.get_group("Israel")["Day of Year"],
             group_by_country.get_group("Israel")["Temp"])
    group_by_country = data[data["Country"] != "Israel"].groupby("Country")
    loss = pd.DataFrame()
    for country in group_by_country:
        loss = loss.append({"Loss": poly.loss(country[1]["Day of Year"],
                                              country[1]["Temp"]),
                            "Country": country[0]}, ignore_index=True)

    fig = px.bar(loss, x="Country", y="Loss")
    fig.show()
