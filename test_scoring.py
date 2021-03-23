from pathlib import Path
from numpy.core.fromnumeric import mean
from numpy.lib.shape_base import _put_along_axis_dispatcher
from numpy.lib.ufunclike import _fix_and_maybe_deprecate_out_named_y
from pandas.io.pytables import AppendableMultiSeriesTable
from sklearn import metrics
import pandas as pd
import numpy as np
from github import Github
import os
from math import floor, log10


def main():
    ak = os.environ.get("SOMEVAR", None)
    g = Github(ak)
    pr_id = os.environ.get("TRAVIS_PULL_REQUEST", None)
    if pr_id == "false" or pr_id == None:
        pr_id = 1
    repo = g.get_repo(350697371)
    pr = repo.get_pull(int(pr_id))
    issue_str = ""
    points_total = 0

    if Path("task_1_predictions.csv").exists():
        df = pd.read_csv("hidden_test_data/test_crystals.csv")
        y_true = np.array(df["calculated_density"])
        y_pred_df = pd.read_csv("task_1_predictions.csv", header=None)
        y_pred = [j for i in y_pred_df.to_numpy() for j in i]
        mae = np.around(metrics.mean_absolute_error(y_true, y_pred), 3)
        r2 = np.around(metrics.r2_score(y_true, y_pred), 3)
        nMAE = np.around(calc_nMAE(y_true, y_pred), 3)
        npoints = points(1 - calc_nMAE(y_true, y_pred))
        points_total += npoints
        issue_str += "Task 1 Prediction - Density\n-----------------\n"
        issue_str += f"Mean Absolute Error: {mae}\n"
        issue_str += f"R<sup>2</sup>: {r2}\n"
        issue_str += f"Normalised Mean Absolute Error (Assessed): {nMAE}\n"
        issue_str += f"__Points: {npoints}__\n\n"
    else:
        issue_str += "Task 1 Prediction - Density\n-----------------\n"
        issue_str += "No results submitted for task 1\n\n"

    if Path("task_2_predictions.csv").exists():
        df = pd.read_csv("hidden_test_data/test_crystals.csv")
        y_true = np.array(df["is_centrosymmetric"])
        y_pred_df = pd.read_csv("task_2_predictions.csv", header=None)
        y_pred = [j for i in y_pred_df.to_numpy() for j in i]
        acc = np.around(metrics.accuracy_score(y_true, y_pred), 3)
        f1mac = np.around(metrics.f1_score(y_true, y_pred, average="macro"), 3)
        f1wei = np.around(metrics.f1_score(y_true, y_pred, average="weighted"), 3)
        npoints = points(metrics.f1_score(y_true, y_pred, average="macro"))
        points_total += npoints
        issue_str += "Task 2 Prediction - Centrosymmetric\n-----------------\n"
        issue_str += f"Accuracy: {acc}\n"
        issue_str += f"Macro F1-score (Assessed): {f1mac}\n"
        issue_str += f"Weighted F1-score: {f1wei}\n"
        issue_str += f"__Points: {npoints}__\n\n"
    else:
        issue_str += "Task 2 Prediction - Centrosymmetric\n-----------------\n"
        issue_str += "No results submitted for task 2\n\n"

    if Path("task_3_predictions.csv").exists():
        df = pd.read_csv("hidden_test_data/test_centroid_distances.csv")
        y_true = np.array(df["mean"])
        y_pred_df = pd.read_csv("task_3_predictions.csv", header=None)
        y_pred = [j for i in y_pred_df.to_numpy() for j in i]
        mae = np.around(metrics.mean_absolute_error(y_true, y_pred), 3)
        r2 = np.around(metrics.r2_score(y_true, y_pred), 3)
        nMAE = np.around(calc_nMAE(y_true, y_pred), 3)
        npoints = points(1 - calc_nMAE(y_true, y_pred))
        points_total += npoints
        issue_str += "Task 3 Prediction - Mean Distances\n-----------------\n"
        issue_str += f"Mean Absolute Error: {mae}\n"
        issue_str += f"R<sup>2</sup>: {r2}\n"
        issue_str += f"Normalised Mean Absolute Error (Assessed): {nMAE}\n"
        issue_str += f"__Points: {npoints}__\n\n"
    else:
        issue_str += "Task 3 Prediction - Mean Distances\n-----------------\n"
        issue_str += "No results submitted for task 3\n\n"

    if Path("task_4_predictions.csv").exists():
        df = pd.read_csv("hidden_test_data/test_distances.csv")
        y_true = np.array(df["n_vdw_contacts"])
        y_pred_df = pd.read_csv("task_3_predictions.csv", header=None)
        y_pred = [j for i in y_pred_df.to_numpy() for j in i]
        mae = np.around(metrics.mean_absolute_error(y_true, y_pred), 3)
        r2 = np.around(metrics.r2_score(y_true, y_pred), 3)
        nMAE = np.around(calc_nMAE(y_true, y_pred), 3)
        npoints = points(1 - calc_nMAE(y_true, y_pred))
        points_total += npoints
        issue_str += "Task 4 Prediction - Van der Waals Contacts\n-----------------\n"
        issue_str += f"Mean Absolute Error: {mae}\n"
        issue_str += f"R<sup>2</sup>: {r2}\n"
        issue_str += f"Normalised Mean Absolute Error (Assessed): {nMAE}\n"
        issue_str += f"__Points: {npoints}__\n\n"
    else:
        issue_str += "Task 4 Prediction - Van der Waals Contacts\n-----------------\n"
        issue_str += "No results submitted for task 4\n\n"

    if Path("bonus_1_predictions.csv").exists():
        df = pd.read_csv("hidden_test_data/test_crystals.csv")
        y_true = np.array(df["packing_coefficient"])
        y_pred_df = pd.read_csv("bonus_1_predictions.csv", header=None)
        y_pred = [j for i in y_pred_df.to_numpy() for j in i]
        mae = np.around(metrics.mean_absolute_error(y_true, y_pred), 3)
        r2 = np.around(metrics.r2_score(y_true, y_pred), 3)
        nMAE = np.around(calc_nMAE(y_true, y_pred), 3)
        npoints = points(1 - calc_nMAE(y_true, y_pred), max_points=10)
        points_total += npoints
        issue_str += "Bonus Task 1 Prediction - Packing Coefficient\n-----------------\n"
        issue_str += f"Mean Absolute Error: {mae}\n"
        issue_str += f"R<sup>2</sup>: {r2}\n"
        issue_str += f"Normalised Mean Absolute Error (Assessed): {nMAE}\n"
        issue_str += f"__Points: {npoints}__\n\n"
    else:
        issue_str += "Bonus Task 1 Prediction - Packing Coefficient\n-----------------\n"
        issue_str += "No results submitted for bonus task 1\n\n"

    if Path("bonus_2_predictions.csv").exists():
        df = pd.read_csv("hidden_test_data/test_crystals.csv")
        y_true = np.array(df["cell_volume"])
        y_pred_df = pd.read_csv("bonus_2_predictions.csv", header=None)
        y_pred = [j for i in y_pred_df.to_numpy() for j in i]
        mae = np.around(metrics.mean_absolute_error(y_true, y_pred), 3)
        r2 = np.around(metrics.r2_score(y_true, y_pred), 3)
        nMAE = np.around(calc_nMAE(y_true, y_pred), 3)
        npoints = points(1 - calc_nMAE(y_true, y_pred), max_points=10)
        points_total += npoints
        issue_str += "Bonus Task 2 Prediction - Cell Volume\n-----------------\n"
        issue_str += f"Mean Absolute Error: {mae}\n"
        issue_str += f"R<sup>2</sup>: {r2}\n"
        issue_str += f"Normalised Mean Absolute Error (Assessed): {nMAE}\n"
        issue_str += f"__Points: {npoints}__\n\n"
    else:
        issue_str += "Bonus Task 2 Prediction - Cell Volume\n-----------------\n"
        issue_str += "No results submitted for bonus task 2\n\n"

    if Path("bonus_3_predictions.csv").exists():
        df = pd.read_csv("hidden_test_data/test_crystals.csv")
        y_true = np.array(df["spacegroup_symbol"])
        y_pred_df = pd.read_csv("bonus_3_predictions.csv", header=None)
        y_pred = [j for i in y_pred_df.to_numpy() for j in i]
        acc = np.around(metrics.accuracy_score(y_true, y_pred), 3)
        f1mac = np.around(metrics.f1_score(y_true, y_pred, average="macro"), 3)
        f1wei = np.aronud(metrics.f1_score(y_true, y_pred, average="weighted"), 3)
        npoints = points(metrics.f1_score(y_true, y_pred, average="macro"), max_points=10)
        points_total += npoints
        issue_str += "Bonus Task 3 Prediction - Space Group Symbol\n-----------------\n"
        issue_str += f"Accuracy: {acc}\n"
        issue_str += f"Macro F1-score (Assessed): {f1mac}\n"
        issue_str += f"Weighted F1-score: {f1wei}\n"
        issue_str += f"__Points: {npoints}__\n\n"
    else:
        issue_str += "Bonus Task 3 Prediction - Space Group Symbol\n-----------------\n"
        issue_str += "No results submitted for bonus task 3\n\n"

    issue_str += "__Total Points__\n-----------------\n"
    issue_str += f"{points_total}\n\n"

    pr.create_issue_comment(issue_str)


def calc_nMAE(true, pred):
    return sum(abs(true - pred)) / sum(abs(true))


def points(score, alpha=1.2, max_points=25):
    return np.minimum(
        (max_points + 2) ** (np.maximum(score, np.finfo(float).eps) ** alpha)
        - 1,
        max_points,
    ).astype(int)


if __name__ == "__main__":
    main()
