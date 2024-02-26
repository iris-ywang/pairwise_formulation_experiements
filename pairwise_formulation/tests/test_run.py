import pandas as pd
import os

from pairwise_formulation.pairwise_data import PairwiseDataInfo
from pairwise_formulation.pairwise_model import PairwiseModel

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression


if __name__ == '__main__':
    data = pd.read_csv("test_data.csv")
    train_set, test_set = train_test_split(data, test_size = 0.1, random_state=40)

    pairwise_data = PairwiseDataInfo(train_set, test_set)
    pairwise_model = PairwiseModel(
        pairwise_data_info=pairwise_data,
        ML_cls=LogisticRegression(),
        ML_reg=LinearRegression()
    ).fit()

    Y_predictions = pairwise_model.predict()
