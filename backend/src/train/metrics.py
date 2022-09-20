"""
Получение метрик и их сохранение
"""

import pandas as pd
import yaml
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)


def get_metrics(
    y_test: np.array, y_bin_test: np.array, y_pred: np.array, y_prob: np.array
):
    """
    функция для сохранения метрик в словарь
    :param y_test: target
    :param y_bin_test: бинаризованный target (1,0)
    :param y_pred: предсказанные target
    :param y_prob: предсказанные вероятности target
    """
    dict_metrics = {
        "roc_auc": round(roc_auc_score(y_bin_test, y_prob), 3),
        "precision_micro": round(precision_score(y_test, y_pred, average="micro"), 3),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro"), 3),
        "recall_micro": round(recall_score(y_test, y_pred, average="micro"), 3),
        "recall_macro": round(recall_score(y_test, y_pred, average="macro"), 3),
        "f1_micro": round(f1_score(y_test, y_pred, average="micro"), 3),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 3),
        "logloss": round(log_loss(y_test, y_prob), 3),
    }
    return dict_metrics


def save_metrics(
    X: pd.DataFrame, y: pd.Series, y_bin: pd.Series, model: object, metrics_path: str
) -> None:
    """
    :param X: матрица объект-признаки
    :param y: целевая переменная
    :param y_bin: бинаризованная целевая переменная
    :param model: выбранный алгоритм
    :param netrics_path: путь сохранения метрик
    """
    result_metrics = get_metrics(
        y_test=y,
        y_bin_test=y_bin,
        y_pred=model.predict(X),
        y_prob=model.predict_proba(X),
    )
    with open(metrics_path, "w") as file:
        json.dump(result_metrics, file)


def load_metrics(config_path: str) -> dict:
    """
    :param config_path: путь до config
    :return: словарь с полученными метриками
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config["train"]["metrics_path"]) as json_file:
        metrics = json.load(json_file)

    return metrics
