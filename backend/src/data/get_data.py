"""
Загрузка данных
"""

import pandas as pd
from typing import Text


def load_dataset(dataset_path: Text) -> pd.DataFrame:
    """
    Получение данных по заданному пути
    :param dataset_path: путь к датасету
    :return: датасет
    """
    return pd.read_csv(dataset_path)
