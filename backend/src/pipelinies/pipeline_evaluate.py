"""
Получение предсказаний классов на основе обученной модели
"""
import os
import yaml
import json
import joblib
import pandas as pd
from ..data.get_data import load_dataset
from ..transform.transform import pipeline_train


def pipeline_evaluating(
    config_path: str, data: pd.DataFrame = None, data_path=None
) -> list:
    """
    :param config_path: путь до config
    :param data: датасет
    :param data_path: путь до датасета
    :return: предсказанные классы
    """
    # подгрузка config
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    preprocess = config["preprocessing"]
    training = config["train"]

    if data_path:
        data = load_dataset(dataset_path=data_path)

    # обработка датасета
    data = pipeline_train(data=data, **preprocess)

    # получение предсказаний
    model = joblib.load(os.path.join(training["model_path"]))
    prediction = model.predict(data).tolist()
    # перевод закодированной переменной в текстовый формат
    with open(preprocess["mapping"]) as f:
        dict_code = json.load(f)
    dict_reverse = {g: i for i, g in dict_code.items()}
    prediction = pd.Series(prediction).map(dict_reverse).tolist()

    return prediction
