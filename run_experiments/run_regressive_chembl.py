import logging

import pandas as pd
import numpy as np
import os
import warnings
import openml

from pairwise_formulation.pa_basics.import_data import \
    filter_data, kfold_splits, transform_categorical_columns, get_repetition_rate
from utils import run
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    chembl_info_all = pd.read_csv(
        "../data/chembl_meta_ml_info.csv"
    )
    chembl_info = chembl_info_all[
        (chembl_info_all["All boolean?"] == False) &
        (chembl_info_all["Half Boolean?"] == False) &
        (chembl_info_all["N(feature)"] > 50) &
        (chembl_info_all["N(sample)"] >= 30)
    ].sort_values(by=["N(sample)"])

    output_dir = "../output/reg_chembl/"

    try:
        existing_results = np.load(output_dir + "regressive_chembl_rf_run1.npy")
        existing_count = len(existing_results)
        all_metrics = list(existing_results)
    except FileNotFoundError:
        existing_results = None
        existing_count = 0
        all_metrics = []

    count = 0
    for file in range(len(chembl_info)):
        count += 1
        if count <= existing_count:
            continue

        data_id = int(chembl_info.iloc[file]["OpenML ID"])
        chembl_id = int(chembl_info.iloc[file]["ChEMBL ID"])
        data = openml.datasets.get_dataset(data_id)
        X, y, categorical_indicator, attribute_names = data.get_data(target=data.default_target_attribute)

        # Exclude datasets with following traits
        if y.nunique() == 1:
            logging.warning(f"Dataset No. {count}, ChEMBL ID {chembl_id}, only has one value of y. Abort.")
            continue

        if get_repetition_rate(np.array([y]).T) >= 0.85:
            logging.warning(f"Dataset No. {count}, ChEMBL ID {chembl_id}, has too many repeated y ( > 85% of y are the same). Abort.")
            continue

        logging.info(f"Running on Dataset No. {count}, ChEMBL ID {chembl_id}, OpenML ID {data_id}")
        train_test = pd.concat([y, X], axis=1)

        col_non_numerical = list(train_test.dtypes[train_test.dtypes == "category"].index) + \
                            list(train_test.dtypes[train_test.dtypes == "object"].index)
        if col_non_numerical:
            train_test = transform_categorical_columns(train_test, col_non_numerical)

        train_test = train_test.to_numpy().astype(np.float64)
        train_test = filter_data(train_test, shuffle_state=1)

        train_test_splits_dict = kfold_splits(train_test=train_test, fold=10)

        metrics_per_dataset = run(
            train_test_splits_dict=train_test_splits_dict,
            ML_cls=RandomForestClassifier(random_state=1, n_jobs=-1),
            ML_reg=RandomForestRegressor(random_state=1, n_jobs=-1),
            percentage_of_top_samples=0.1,  # top-performing as in top 10%
        )
        all_metrics.append(metrics_per_dataset)
        np.save(output_dir + "regressive_chembl_rf_run1.npy", np.array(all_metrics))
