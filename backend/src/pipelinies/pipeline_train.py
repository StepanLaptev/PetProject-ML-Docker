"""
Объединение функций для обучения модели
"""
import os
import joblib
import yaml
from ..data.split_dataset import *
from ..train.train import find_best_parametres, train_model
from ..data.get_data import load_dataset
from ..transform.transform import pipeline_train


def pipeline_training(config_path: str) -> None:
    """
    :param config_path: путь до config
    :return: None
    """
    # загрузка config
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocess = config["preprocessing"]
    training = config["train"]

    # подгрузка данных
    df = load_dataset(dataset_path=preprocess["train_path"])

    # обработка датасета
    df = pipeline_train(data=df, **preprocess)

    # разбиение на train и test
    X_train, X_test, y_train, y_bin_train, y_test, y_bin_test = split_data(
        df, **preprocess
    )

    # подбор наилучших параметров
    study = find_best_parametres(X_train=X_train, y_train=y_train, **training)

    # обучение модели с подобранными параметрами
    clf_LGBM = train_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_bin_test=y_bin_test,
        study=study,
        metric_path=training["metrics_path"],
    )

    # сохранение результатов: модель + параметры
    joblib.dump(clf_LGBM, os.path.join(training["model_path"]))
    joblib.dump(study, os.path.join(training["study_path"]))
