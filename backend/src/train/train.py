'''
Обучение и подбор параметров
'''

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
from ..data.split_dataset import get_eval_set
from ..train.metrics import save_metrics

from lightgbm import LGBMClassifier
import optuna
from optuna import Study


def objective_lgbm(trial, X: pd.DataFrame, y: pd.Series, N_FOLDS: int, random_state=20):
    '''
    :param trial: кол-во итераций
    :param X: датасет объект-признак
    :param y: датасет target
    :param N_FOLDS: кол-во разбиений датасета
    :param random_state: фиксация значений
    :return: значение метрики logloss
    '''
    parameters = {
        "n_estimators": trial.suggest_categorical("n_estimators", [1500]),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 5, 20, step=2),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 30, 60, step=2),
        "max_depth": trial.suggest_int("max_depth", 2, 4, step=1),
        "boosting_type": trial.suggest_categorical(
            "boosting_type", ["gbdt", "goss", "dart"]
        ),
        "bagging_fraction": trial.suggest_categorical(
            "bagging_fraction", [0.6, 0.7, 0.8, 0.9]
        ),
        "feature_fraction": trial.suggest_categorical(
            "feature_fraction", [0.6, 0.7, 0.8, 0.9]
        ),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 50),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 50),
        "is_unbalance": trial.suggest_categorical("is_unbalance", [True]),
        "random_state": random_state,
        "objective": trial.suggest_categorical("objective", ["multiclass"]),
        "num_class": trial.suggest_categorical("num_class", [6]),
        "metric": trial.suggest_categorical("metric", ["multi_logloss"]),
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=15)
    predicts = np.empty(N_FOLDS)

    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        lgbm_class = LGBMClassifier(**parameters)

        lgbm_class.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="multi_logloss",
            early_stopping_rounds=20,
            verbose=0,
        )

        y_score = lgbm_class.predict_proba(X_test)

        predicts[idx] = log_loss(y_test, y_score)

    return np.mean(predicts)


def find_best_parametres(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> Study:
    """
    :param data: датасет
    :param kwargs: параметры
    :return: Study
    """
    study_lgbm = optuna.create_study(direction="minimize", study_name="lgbm")
    func_lgbm = lambda trial: objective_lgbm(
        trial,
        X_train,
        y_train,
        N_FOLDS=kwargs["n_folds"],
        random_state=kwargs["random_state"],
    )
    study_lgbm.optimize(func_lgbm, n_trials=kwargs["n_trials"], show_progress_bar=True)
    return study_lgbm


def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_bin_test: np.array,
    study: Study,
    metric_path: str,
) -> LGBMClassifier:
    """
    :param X_train: матрица объект-признак train
    :param X_test: матрица объект-признак test
    :param y_train: целевая переменная train
    :param y_test: целевая переменная test
    :param y_bin_test: бинаризованный target
    :param study: optuna
    :param metric_path: путь сохранения метрик
    :return: LGBMClassifier
    """
    alf_LGBM = LGBMClassifier(**study.best_params)
    # получение eval_set
    X_train_, y_train_, eval_set = get_eval_set(X_train, y_train)
    # обучение модлеи
    alf_LGBM.fit(X=X_train_, y=y_train_, eval_set=eval_set, early_stopping_rounds=20)

    # сохранение метрик
    save_metrics(
        X=X_test, y=y_test, y_bin=y_bin_test, model=alf_LGBM, metrics_path=metric_path
    )
    return alf_LGBM
