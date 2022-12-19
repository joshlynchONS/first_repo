from keras.models import Sequential
from keras.layers import Dense


def build_model(input_shape):
    """a simple MLP model to predict binary labels of a dataset
    Args:
        input_shape (tuple): the shape of the data used in this model

    Returns:
        keras model: compiled MLP keras modelllll
    """

    model = Sequential()
    model.add(Dense(12, input_shape=(input_shape,), activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model
