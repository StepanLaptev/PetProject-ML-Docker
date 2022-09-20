"""
Предобработка данных
"""

import json
import pandas as pd
import yaml
import warnings
warnings.filterwarnings("ignore")


def lower_case(data: pd.DataFrame, columns: list) -> None:
    """
    приведение к одному регистру
    :param data: датасет
    :param columns: признаки, значения которых необходимо привести к одному регистру
    """
    for i in columns:
        data[i] = data[i].str.lower()


def correct_mistake(data: pd.DataFrame, map_dict_mistake: dict) -> pd.DataFrame:
    """
    Замена значений в датасете
    :param data: датасет
    :param map_dict_mistake: словарь с признаками и значениями
    :return: датасет
    """
    return data.replace(map_dict_mistake)


def save_unique_data(
    data: pd.DataFrame, drop_columns: list, target_column: str, unique_values_path: str
) -> None:
    """
    Сохранение словаря с признаками и уникальными значениями
    :param drop_columns: список столбцов/признаков для удаления
    :param data: датасет
    :param target_column: cost_category - target
    :param unique_values_path: путь до файла со словарем
    :return: None
    """
    unique_df = data.drop(
        columns=drop_columns + [target_column], axis=1, errors="ignore"
    )
    # создаю словарь с уникальными значениями для вывода в UI
    # исключая Tour_id и cost_category
    dict_unique = {key: unique_df[key].unique().tolist() for key in unique_df.columns}
    with open(unique_values_path, "w") as file:
        json.dump(dict_unique, file)


def filling_in_gaps(data: pd.DataFrame) -> None:
    """
    заполнение пропусков
    :param data: датасет
    """
    list_data_columns = (
        data.isna().sum().reset_index().rename(columns={"index": "col_name", 0: "cnt"})
    )
    list_columns_zero = list(
        (list_data_columns[list_data_columns.iloc[:, 1] != 0]["col_name"])
    )
    for i in list_columns_zero:
        if data[i].dtype == "int" or data[i].dtype == "float":
            data[i].fillna(data[i].median(), inplace=True)
        else:
            data[i].fillna("None", inplace=True)


def func_map_values(data: pd.DataFrame, names_cols: dict) -> pd.DataFrame:
    """
    замена значений в датасете
    :param data: датасет
    :param names_cols: список признаков
    """
    return data.replace(names_cols)


def symbol_correct(data: pd.DataFrame, names_cols: list) -> None:
    """
    замена пробелов и запятых символом "_"
    :param data: датасет
    :param names_cols: список признаков, в значениях которых встречаются пробелы и запятые
    """
    for col in names_cols:
        data[col].replace(" ", "_", regex=True, inplace=True)
        data[col].replace(",", "", regex=True, inplace=True)


def correct_types(data: pd.DataFrame, change_type_columns: dict):
    """
    Преобразование признаков в заданный тип данных
    :param data: датасет
    :param change_type_columns: словарь с признаками и типами данных
    :return:
    """
    return data.astype(change_type_columns, errors="raise")


def del_some_colms(data: pd.DataFrame, name_cols: list) -> None:
    """
    удаление ненужных признаков
    :param data: датасет
    :param name_cols: список признаков
    :return: датасет
    """
    data.drop(columns=name_cols, axis=1, inplace=True)


def pipeline_train(
    data: pd.DataFrame, flag_evaluate: bool = True, **kwargs
) -> pd.DataFrame:
    """
    пайплайн для предобработки данных
    :param data: датасет
    :param kwargs: параметры
    :return: pd.DataFrame
    """
    # приведение к одному регистру
    lower_case(data=data, columns=kwargs["lower_columns"])
    # исправление орф.ошибок в датасете
    data = correct_mistake(data=data, map_dict_mistake=kwargs["correct_values"])
    # Сохранение словаря с признаками и уникальными значениями
    # при вводе в UI не происходит перезапись уникальных значений
    if flag_evaluate == False:
        save_unique_data(
            data=data,
            drop_columns=kwargs["drop_column"],
            target_column=kwargs["target"],
            unique_values_path=kwargs["unique_values_path"],
        )
    # заполнение пропусков
    filling_in_gaps(data=data)
    # замена значений в датасете
    data = func_map_values(data=data, names_cols=kwargs["map_func_for_columns"])
    # замена пробелов и запятых символом "_"
    symbol_correct(data=data, names_cols=kwargs["columns_symbol_correct"])
    # преобразование некоторых признаков в тип category
    data = correct_types(data=data, change_type_columns=kwargs["change_type_columns"])
    # удаление признаков
    if kwargs["drop_column"][0] in data.columns.tolist():
        del_some_colms(data=data, name_cols=kwargs["drop_column"])

    return data
