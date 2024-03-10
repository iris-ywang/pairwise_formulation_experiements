
import pandas as pd
import numpy as np
import warnings
from itertools import product

from run_utils import run_per_stock_dataset
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

warnings.filterwarnings("ignore")


def get_stock_data(folder_path, prefix, learning_year, target_value_column_name):
    year1_df = pd.read_csv(folder_path + f"/{prefix}_{learning_year}.csv", index_col=0)
    year2_df = pd.read_csv(folder_path + f"/{prefix}_{learning_year + 1}.csv", index_col=0)
    year3_df = pd.read_csv(folder_path + f"/{prefix}_{learning_year + 2}.csv", index_col=0)

    train_df = year1_df.merge(year2_df[target_value_column_name], how="left", left_index=True, right_index=True)
    train_df[target_value_column_name + '_x'] = train_df[target_value_column_name + '_y']
    train_df = train_df.drop([target_value_column_name + '_y'], axis=1).rename(
        columns={target_value_column_name + '_x': target_value_column_name}).reset_index()

    test_df = year2_df.merge(year3_df[target_value_column_name], how="left", left_index=True, right_index=True)
    test_df[target_value_column_name + '_x'] = test_df[target_value_column_name + '_y']
    test_df = test_df.drop([target_value_column_name + '_y'], axis=1).rename(
        columns={target_value_column_name + '_x': target_value_column_name}).reset_index()

    return train_df, test_df


if __name__ == '__main__':

    output_dir = "../output/stock_data/"

    n_portofolios = [10, 50]
    years = list(range(2010, 2018))
    random_states = [111,222,333]

    try:
        existing_results = np.load(output_dir + "stock_data_run1.npy")
        existing_count = len(existing_results)
        all_metrics = list(existing_results)
    except FileNotFoundError:
        existing_results = None
        existing_count = 0
        all_metrics = []

    count = 0
    for n_portofolio, year, rs in product(n_portofolios, years, random_states):
        count += 1
        if count <= existing_count:
            continue

        train_df, test_df = get_stock_data(
            folder_path="../data/stock_yearly_data/",
            prefix="stock_data",
            target_value_column_name="annual_pc_price_change",
        )

        train_test_dict = {
            "train_set": train_df,
            "test_set": test_df,
        }

        pred_return_list = run_per_stock_dataset(
            foldwise_data=train_test_dict,
            ML_cls=RandomForestClassifier(random_state=rs, n_jobs=-1),
            ML_reg=RandomForestRegressor(random_state=rs, n_jobs=-1),
            n_portofolio=n_portofolio,  # top-performing as in top 10%
            target_value_col_name="annual_pc_price_change",
        )
        pred_return_list = [n_portofolio, year, rs] + pred_return_list
        all_metrics.append(pred_return_list)
        np.save(output_dir + "stock_data_run1.npy", np.array(all_metrics))
