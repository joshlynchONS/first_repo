import numpy as np


def calc_accuracy(predicted_test, labels_test):
    """summary of functions

    Args:
        predicted_test (numpy array): predicted values
        labels_test (umpy array): actual valuesssss

    Returns:
        float: the accuracy between two arrays displayed as a percentage
    """

    rounded_prediction = np.around(predicted_test, decimals=0).astype(int)

    predicted_vs_labels_arr = np.equal(rounded_prediction, labels_test)

    number_true = np.count_nonzero(predicted_vs_labels_arr)

    percentage_accuracy = (number_true / len(predicted_vs_labels_arr)) * 100

    return percentage_accuracy
