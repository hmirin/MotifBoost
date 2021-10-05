import datetime
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve

human_amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

basic_void_mark = "@"


def get_current_datetime() -> str:
    return datetime.datetime.now().strftime("%y%m%d%H%M%S")


def hdf5dataset_to_pyobject(x):
    data = np.atleast_1d(x)[0]
    if isinstance(data, bytes):
        return data.decode("ascii")
    else:
        return data


def summarize_metrics(
    data_labels: List[str],
    ground_truth_class: List[bool],
    predicted_class: List[bool],
    predicted_proba: Union[List[float], np.ndarray],
    folder_path: str,
    prefix: str,
    mlflow_experiment_name: str,
    weighted: List[float] = None,
):
    prefix = get_current_datetime() + "_" + prefix
    matrix = confusion_matrix(
        [int(i) for i in ground_truth_class],
        [int(i) for i in predicted_class],
    )
    print("confusion matrix")
    print(matrix)
    tn, fp, fn, tp = matrix.ravel()
    print("prec:", tp / (tp + fp), "recall:", tp / (tp + fn))
    fpr, tpr, _ = roc_curve(
        ground_truth_class, predicted_proba, sample_weight=weighted)
    print(ground_truth_class)
    print(predicted_proba)
    roc_auc = auc(fpr, tpr)
    print("auc:", roc_auc)
    lw = 2
    plt.clf()
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" %
        roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC/AUC for disease recognition")
    plt.legend(loc="lower right")
    save_fig_path = (
        folder_path + "/" + prefix + "_roc_auc_" + "%0.2f" % roc_auc + ".png"
    )
    plt.savefig(save_fig_path)
    excel_path = folder_path + "/" + prefix + "_experiment_log.xlsx"
    pd.DataFrame.from_dict(
        {
            "ground_truth": ground_truth_class,
            "predicted_class": predicted_class,
            "predicted_proba": predicted_proba,
            "data_labels": data_labels,
        }
    ).to_excel(excel_path)
    mlflow.set_experiment(mlflow_experiment_name)
    with mlflow.start_run(run_name=prefix):
        mlflow.log_artifact(save_fig_path)
        mlflow.log_artifact(excel_path)
        mlflow.log_metrics({"tp": tp, "fn": fn, "fp": fp,
                           "tn": tn, "roc_auc": roc_auc})
    return roc_auc


def get_saved_files_path(
    save_dir: str, experiment_id: str, sample_id: str
) -> Tuple[Path, Path]:
    path = Path(save_dir + "/" + experiment_id + "/" + sample_id)
    return path / "sequence_data.feather", path / "info.pkl"
