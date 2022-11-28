from src.make_data.read_data import split_data
from src.make_models.initial_model import build_model
from src.utils.utils import calc_accuracy
import numpy as np

# ---- Config file ----#
train_dir = "D:/Repos/first_repo/data/raw/archive/train.csv"
column_headers = [
    "age_of_car",
    "age_of_policyholder",
    "is_claim",
    "make",
    "is_central_locking",
]
# ------------------- #

features_train, labels_train, features_test, labels_test = split_data(
    train_dir, column_headers
)

model = build_model(np.shape(features_train)[1])
model.fit(features_train, labels_train, epochs=3, batch_size=10)

predicted_test = model.predict(features_test)

accuracy = calc_accuracy(predicted_test, labels_test)

print(accuracy)
