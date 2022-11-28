import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(dir, column_headers):
    """reads a csv daatset given it's directory

    Args:
        dir (string): directory of csv dataset
        column_headers (list of strings): list of column headers to read in from dataset

    Returns:
        pandas df: dataframe of data in directory
    """

    df = pd.read_csv(dir, usecols=column_headers)

    return df


def split_data(dir, column_headers):
    """reating a test/train split on a csv dataset

    Args:
        dir (string): directory of data (csv format) to test/train split
        column_headers (list of strings): list of column headers to read in from dataset

    Returns:
        numpy array: 4 numpy arrays of a test/train split on data
    """

    df = read_data(dir, column_headers)

    train, test = train_test_split(df, test_size=0.2)

    features_train = train[["age_of_car", "age_of_policyholder"]].to_numpy()
    features_test = test[["age_of_car", "age_of_policyholder"]].to_numpy()

    lables_train = train[["is_claim"]].to_numpy()
    lables_test = test[["is_claim"]].to_numpy()

    return features_train, lables_train, features_test, lables_test
