import pickle

import matplotlib
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import argparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.db import connect


def load_data(ase_db_path, target="S1_exc"):
    # load data from ase db

    # Load representations, labels and values
    x_pool = []
    y_pool = []
    with connect(ase_db_path) as db:
        for db_row in db.select():
            x_pool.append(db_row.data["slatm"])
            y_pool.append(db_row.data[target])
    x_pool = np.array(x_pool)
    y_pool = np.array(y_pool)
    return x_pool, y_pool


def fws_scaling(x_pool):
    left_shape = x_pool.shape[1] - 13 - 4368
    ic = []
    one_b = list(range(0, 13))
    ic.append(one_b)
    two_b = list(range(13, 4368 + 13))
    ic.append(two_b)
    three_b = list(range(4368 + 13, left_shape + 4368 + 13))
    ic.append(three_b)

    fw_one_b = 1 / (3 * 13)
    fw_two_b = 1 / (3 * 4368)
    fw_three_b = 1 / (3 * left_shape)
    fws = np.zeros((x_pool.shape[1]))
    for i in range(x_pool.shape[1]):
        if i < 13:
            fws[i] = fw_one_b
        elif i < 4381:
            fws[i] = fw_two_b
        else:
            fws[i] = fw_three_b
    assert np.isclose(np.sum(fws), 1)
    return fws


def plot_learning_curves(results, target,save_path):
    # Plot learning curves
    plt.rc("font", size=22)
    plt.plot(results["validation_0"]["mae"], label="train")
    plt.plot(results["validation_1"]["mae"], label="test")
    plt.xlabel("Estimators")
    plt.ylabel(f"{target} MAE (eV)$")
    # Show the legend
    plt.legend()
    # Show and save the plot
    plt.show()
    plt.savefig(save_path +"/xgboost_S1_exc_lc.png")


def main(target, dataset_path, save_path):
    # Load data
    x_pool, y_pool = load_data(dataset_path, target)
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        x_pool, y_pool, test_size=0.1, random_state=1
    )
    fws = fws_scaling(X_train)
    # Define the model
    model = XGBRegressor(
        n_estimators=5000,
        eta=0.05,
        colsample_bytree=0.75,
        max_depth=8,
        eval_metric="mae",
        early_stopping_rounds=10,
    )
    # Define the datasets to evaluate each iteration for early stopping
    evalset = [(X_train, y_train), (X_test, y_test)]

    # Fit the model
    model.fit(
        X_train,
        y_train,
        eval_set=evalset,
        feature_weights=fws,
    )
    # Make predictions
    y_pred = model.predict(X_test)
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("MAE: ", mae)
    print("R2: ", r2)
    # Save the model
    with Path(save_path, f"model_{target}.pkl").open("wb") as f:
        pickle.dump(model, f)

    # Retrieve performance metrics
    results = model.evals_result()

    # Plot learning curves
    plot_learning_curves(results, target,save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost model on SLATM features"
    )
    parser.add_argument("--target", type=str, default="S1_exc", help="Target property")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/media/mohammed/Work/FORMED_ML/ase_database/formed.db",
        help="Path to the ASE database",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/media/mohammed/Work/FORMED_ML/models/xgboost_slatm",
        help="Path to save the model",
    )
    args = parser.parse_args()
    main(args.target, args.dataset_path,    args.save_path )
