from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
from IMLearn.base.base_estimator import  BaseEstimator
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(filename).dropna().drop_duplicates()
    full_data["booking_datetime"] = pd.to_datetime(full_data["booking_datetime"])
    full_data["checkin_date"] = pd.to_datetime(full_data["checkin_date"])
    full_data["checkout_date"] = pd.to_datetime(full_data["checkout_date"])
    full_data["hotel_live_date"] = pd.to_datetime(full_data["hotel_live_date"])
    full_data["cancellation_datetime"] = pd.to_datetime(
        full_data["cancellation_datetime"])

    full_data["data_diff"] = (full_data["cancellation_datetime"] - full_data[
        "booking_datetime"]).astype('timedelta64[D]')
    full_data["time_till_booking"] = (
                full_data["cancellation_datetime"] - full_data[
            "booking_datetime"]).astype('timedelta64[D]')
    full_data["booking_datetime_m"] = full_data["booking_datetime"].dt.to_period(
        "M")
    full_data["checkin_date_m"] = (full_data["checkin_date"]).dt.to_period("M")
    full_data.head()
    features = full_data[["h_booking_id",
                          "hotel_id",
                          "accommadation_type_name",
                          "hotel_star_rating",
                          "customer_nationality","hotel_country_code",
                          # "booking_datetime","checkin_date","checkout_date",
                          # "hotel_live_date","cancellation_datetime",
                          "data_diff","time_till_booking",
                          "booking_datetime_m","checkin_date_m"]]

    features["customer_nationality"]  = change_country(features["customer_nationality"])
    features["hotel_country_code"] = change_country(
        features["hotel_country_code"])
    features["holiday_in_my_country"] = \
        features["customer_nationality"]==features["hotel_country_code"]
    features["booking_datetime_m"] = features["booking_datetime_m"].apply(str)
    features["checkin_date_m"] = features["checkin_date_m"].apply(str)
    features = pd.get_dummies(features)

    labels = full_data["cancellation_datetime"]

    return features, labels


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pred = estimator.predict(X)
    pd.DataFrame(pred, columns=["predicted_values"]).to_csv(filename,
                                                          index=False)


def find_country(country:str,country_data:pd.DataFrame)->str:
    for col in country_data.columns:
        try:
            filtered = pd.Index(country_data[col]).get_loc(country)
        except KeyError:
            continue
        return country_data["name_short"][filtered]


def change_country(col:pd.Series)->pd.DataFrame:
    country_base = pd.read_csv("country_data.tsv",sep="\t",header=0)
    return col.apply(lambda X:find_country(X,country_base))


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)

    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(train_X, train_y)

    # Store model predictions over test set
    evaluate_and_export(estimator, test_X, "id1_id2_id3.csv")

    pred = estimator.predict(test_X)
    print()
