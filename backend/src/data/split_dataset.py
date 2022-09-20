"""
Разделение датасета на train и test
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from sklearn.preprocessing import label_binarize, LabelEncoder
import json


def split_data(
    data: pd.DataFrame, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Получение train и test датасетов
    :param data: датасет
    :param kwargs: параметры
    :return: набор данных train/test + eval_set
    """
    # кодировка target для повышения параметра ROC_AUC
    enc = LabelEncoder()
    data[kwargs["target"]] = enc.fit_transform(data[kwargs["target"]])
    # словарь кат.значение - код.значение
    mapping = dict(zip(enc.classes_, range(len(enc.classes_))))
    # сохранение словаря
    with open(kwargs['mapping'], "w") as file:
        json.dump(mapping, file)

    X = data.drop(columns=[kwargs["target"]], axis=1)
    y = data[kwargs["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=kwargs["test_size"],
        random_state=kwargs["random_state"],
    )

    y_bin_train = label_binarize(y_train, classes=list(set(y)))
    y_bin_test = label_binarize(y_test, classes=list(set(y)))

    return X_train, X_test, y_train, y_bin_train, y_test, y_bin_test


def get_eval_set(X: pd.DataFrame, y: pd.Series):
    """
    получение eval_set
    :param X: матрица объект-признак
    :param y: целевая переменная
    :return: eval set для обучения
    """
    X_train_, X_val, y_train_, y_val = train_test_split(
        X, y, test_size=0.16, shuffle=True, random_state=15
    )
    eval_set = [(X_val, y_val)]
    return X_train_, y_train_, eval_set
