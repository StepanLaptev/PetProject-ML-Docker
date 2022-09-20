"""
Тренировка модели + вывод метрик и графиков
"""
import os
import json
import joblib
import requests
import streamlit as st
from optuna.visualization import plot_param_importances, plot_optimization_history


def start_training(config: dict, endpoint: object) -> None:
    """
    Тренировка модели с выводом результатов
    :param config: конфигурационный файл
    :param endpoint: endpoint
    """
    # Last metrics
    if os.path.exists(config["train"]["metrics_path"]):
        with open(config["train"]["metrics_path"]) as json_file:
            old_metrics = json.load(json_file)
    else:
        # если модель ранее не обучалась
        old_metrics = {
            "roc_auc": 0,
            "precision_micro": 0,
            "precision_macro": 0,
            "recall_micro": 0,
            "recall_macro": 0,
            "f1_micro": 0,
            "f1_macro": 0,
            "logloss": 0,
        }

    # Train
    with st.spinner("Идет обучение модели. Пожалуйста, подождите..."):
        output = requests.post(endpoint, timeout=8000)
    st.success("Success!")

    new_metrics = output.json()["metrics"]

    # diff metrics
    (
        roc_auc,
        precision_micro,
        precision_macro,
        recall_micro,
        recall_macro,
        f1_micro,
        f1_macro,
        logloss,
    ) = st.columns(8)

    roc_auc.metric(
        "ROC-AUC",
        round(new_metrics["roc_auc"], 2),
        f"{new_metrics['roc_auc']-old_metrics['roc_auc']:.2f}",
    )
    precision_micro.metric(
        "Precision micro",
        round(new_metrics["precision_micro"], 2),
        f"{new_metrics['precision_micro']-old_metrics['precision_micro']:.2f}",
    )

    precision_macro.metric(
        "Precision macro",
        round(new_metrics["precision_macro"], 2),
        f"{new_metrics['precision_macro'] - old_metrics['precision_macro']:.2f}",
    )

    recall_micro.metric(
        "Recall micro",
        round(new_metrics["recall_micro"], 2),
        f"{new_metrics['recall_micro']-old_metrics['recall_micro']:.2f}",
    )

    recall_macro.metric(
        "Recall macro",
        round(new_metrics["recall_macro"], 2),
        f"{new_metrics['recall_macro'] - old_metrics['recall_macro']:.2f}",
    )

    f1_micro.metric(
        "F1 micro",
        round(new_metrics["f1_micro"], 2),
        f"{new_metrics['f1_micro']-old_metrics['f1_micro']:.2f}",
    )

    f1_macro.metric(
        "F1 macro",
        round(new_metrics["f1_macro"], 2),
        f"{new_metrics['f1_macro'] - old_metrics['f1_macro']:.2f}",
    )

    logloss.metric(
        "Logloss",
        round(new_metrics["logloss"], 2),
        f"{new_metrics['logloss']-old_metrics['logloss']:.2f}",
    )

    # plot study
    study = joblib.load(os.path.join(config["train"]["study_path"]))
    fig_imp = plot_param_importances(study)
    fig_history = plot_optimization_history(study)

    st.plotly_chart(fig_imp, use_container_width=True)
    st.plotly_chart(fig_history, use_container_width=True)
