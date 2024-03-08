"""
import ChEMBL dataset and convert it to filtered np.array
"""

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold



def dataset(filename, shuffle_state=None):
    orig_data = import_data(filename, shuffle_state)
    filter1 = uniform_features(orig_data)
    filter2 = duplicated_features(filter1)
    return filter2


def import_data(filename, shuffle_state):
    """
    to import raw data from csv to numpy array
    :param filename: str - string of (directory of the dataset + filename)
    :param shuffle_state: int - random seed
    :return: np.array - [y, x1, x2, ..., xn]
    """
    df = pd.read_csv(filename)
    try:
        data = pd.DataFrame(data=df).to_numpy().astype(np.float64)
    except:
        del df['molecule_id']
        data = pd.DataFrame(data=df).to_numpy().astype(np.float64)

    if shuffle_state is not None:
        data = shuffle(data, random_state=shuffle_state)
    return data


def uniform_features(data):
    """
    Remove the all-zeros and all-ones features from the dataset
    :param data: np.array - [y, x1, x2, ..., xn]
    :return: np.array - [y, x1, x2, ..., xn']
    """
    n_samples, n_columns = np.shape(data)
    redundant_features = []
    for feature in range(1, n_columns):
        if np.all(data[:, feature] == data[0, feature]):
            redundant_features.append(feature)
    filtered_data = np.delete(data, redundant_features, axis=1)
    return filtered_data


def duplicated_features(data):
    """
    Remove the duplicated features from the dataset
    **Note: this function can change the sequences of features

    :param data: np.array - [y, x1, x2, ..., xn]
    :return: np.array - [y, x1, x2, ..., xn']
    """
    y = data[:, 0:1]
    a = data[:, 1:].T
    order = np.lexsort(a.T)
    a = a[order]

    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    new_train = a[ui].T
    new_train_test = np.concatenate((y, new_train), axis=1)
    return new_train_test


def kfold_splits(train_test: np.array, fold=10, random_state=None) -> dict:
    train_test_data = {}
    kf = KFold(n_splits=fold, shuffle=True, random_state=random_state)
    n_fold = 0
    for train_ids, test_ids in kf.split(train_test):
        train_test_data_per_fold = {'train_set': train_test[train_ids], 'test_set': train_test[test_ids]}
        train_test_data[n_fold] = train_test_data_per_fold
        n_fold += 1
    return train_test_data