import logging

from itertools import product
import pandas as pd
import numpy as np
import os
import warnings

from pairwise_formulation.pa_basics.import_data import \
    filter_data, kfold_splits, transform_categorical_columns
from run_utils import run
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    input_dir = "../data/material_discovery_data/"
    output_dir = "../output/material_discovery_data/"
    list_filename = [
        "icsd_formation_energy.csv",
        "mp_band_gap.csv",
    ]

    random_states = list(range(0, 10))
    sizes = [50, 100, 200, 300, 400, 500]

    for filename in list_filename:
        whole_dataset = pd.read_csv(input_dir + filename)
        dataset_name = filename.split(".")[0]

        try:
            existing_results = np.load(output_dir + f"{dataset_name}_rf_run1.npy")
            existing_count = len(existing_results)
            all_metrics = list(existing_results)
        except FileNotFoundError:
            existing_results = None
            existing_count = 0
            all_metrics = []

        count = 0
        for rs, n_samples in product(random_states, sizes):
            count += 1
            if count <= existing_count:
                continue

            logging.info(f"Running dataset {filename}, size {n_samples}, sampling random state {rs}.")

            train_test = whole_dataset.sample(n=n_samples, random_state=rs)

            col_non_numerical = list(train_test.dtypes[train_test.dtypes == "category"].index) + \
                                list(train_test.dtypes[train_test.dtypes == "object"].index)
            if col_non_numerical:
                train_test = transform_categorical_columns(train_test, col_non_numerical)

            train_test = train_test.to_numpy().astype(np.float64)
            train_test = filter_data(train_test, shuffle_state=1)

            if len(np.unique(train_test[:, 0])) == 1:
                logging.warning("Cannot build model with only one target value for Dataset " + filename)
                logging.warning(f"Skip Dataset {filename}")
                continue

            train_test_splits_dict = kfold_splits(train_test=train_test, fold=10)

            metrics_per_dataset = run(
                train_test_splits_dict=train_test_splits_dict,
                ML_cls=RandomForestClassifier(random_state=1, n_jobs=-1),
                ML_reg=RandomForestRegressor(random_state=1, n_jobs=-1),
                percentage_of_top_samples=0.1,  # top-performing as in top 10%
            )
            all_metrics.append(metrics_per_dataset)
            np.save(output_dir + f"{dataset_name}_rf_run1.npy", np.array(all_metrics))
