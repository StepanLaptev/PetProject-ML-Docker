"""
Отображение кнопок для ввода значений
"""
import json
from io import BytesIO
import pandas as pd
import requests
import streamlit as st


def evaluate_input(unique_data_path: str, endpoint: object) -> None:
    """
    Получение входных данных путем ввода в UI -> вывод результата
    :param unique_data_path: путь до уникальных значений
    :param endpoint: endpoint
    :return: None
    """
    with open(unique_data_path) as file:
        unique_df = json.load(file)

    # кнопки
    country = st.sidebar.selectbox("Country", (unique_df["country"]))
    age_group = st.sidebar.selectbox("Age", (unique_df["age_group"]))
    travel_with = st.sidebar.selectbox("Travel with", (unique_df["travel_with"]))
    total_female = st.sidebar.slider("Total female", min_value=0, max_value=50)
    total_male = st.sidebar.slider("Total male", min_value=0, max_value=50)
    purpose = st.sidebar.selectbox("Purpose", (unique_df["purpose"]))
    main_activity = st.sidebar.selectbox("Main activity", (unique_df["main_activity"]))
    info_source = st.sidebar.selectbox("Info source", (unique_df["info_source"]))
    tour_arrangement = st.sidebar.selectbox(
        "Tour arrangement", (unique_df["tour_arrangement"])
    )
    package_transport_int = st.sidebar.radio("Package transport int", ["Yes", "No"])
    package_accomodation = st.sidebar.radio("Package accomodation", ["Yes", "No"])
    package_food = st.sidebar.radio("Package food", ["Yes", "No"])
    package_transport_tz = st.sidebar.radio("Package transport Tanzania", ["Yes", "No"])
    package_sightseeing = st.sidebar.radio("Package sightseeing", ["Yes", "No"])
    package_guided_tour = st.sidebar.radio("Package guided tour", ["Yes", "No"])
    package_insurance = st.sidebar.radio("Package insurance", ["Yes", "No"])
    night_mainland = st.sidebar.slider("Night mainland", min_value=0, max_value=365)
    night_zanzibar = st.sidebar.slider("Night Zanzibar", min_value=0, max_value=365)
    first_trip_tz = st.sidebar.radio("First trip", ["Yes", "No"])

    dict_data = {
        "country": country,
        "age_group": age_group,
        "travel_with": travel_with,
        "total_female": total_female,
        "total_male": total_male,
        "purpose": purpose,
        "main_activity": main_activity,
        "info_source": info_source,
        "tour_arrangement": tour_arrangement,
        "package_transport_int": package_transport_int,
        "package_accomodation": package_accomodation,
        "package_food": package_food,
        "package_transport_tz": package_transport_tz,
        "package_sightseeing": package_sightseeing,
        "package_guided_tour": package_guided_tour,
        "package_insurance": package_insurance,
        "night_mainland": night_mainland,
        "night_zanzibar": night_zanzibar,
        "first_trip_tz": first_trip_tz,
    }

    st.write(
        f"""### Данные туристической группы:\n
    1) Country: {dict_data['country']}
    2) Age: {dict_data['age_group']}
    3) Travel with: {dict_data['travel_with']}
    4) Total female: {dict_data['total_female']}
    5) Total male: {dict_data['total_male']}
    6) Purpose: {dict_data['purpose']}
    7) Main activity: {dict_data['main_activity']}
    9) Info source: {dict_data['info_source']}
    10) Tour arrangement: {dict_data['tour_arrangement']}
    11) Package transport int: {dict_data['package_transport_int']}
    12) Package accomodation: {dict_data['package_accomodation']}
    13) Package food: {dict_data['package_food']}
    14) Package transport tz: {dict_data['package_transport_tz']}
    15) Package sightseeing: {dict_data['package_sightseeing']}
    16) Package guided tour: {dict_data['package_guided_tour']}
    17) Package insurance: {dict_data['package_insurance']}
    18) Night mainland: {dict_data['night_mainland']}
    19) Night Zanzibar: {dict_data['night_zanzibar']}
    20) First trip: {dict_data['first_trip_tz']}
        """
    )

    button_ok = st.button("Predict")
    if button_ok:
        result = requests.post(endpoint, timeout=8000, json=dict_data)
        json_str = json.dumps(result.json())
        output = json.loads(json_str)
        st.write(f"## {output}")
        st.success("Success!")


def evaluate_from_file(data: pd.DataFrame, endpoint: object, files: BytesIO):
    """
    Получение входных данных в качестве файла -> вывод результата в виде таблицы
    :param data: датасет
    :param endpoint: endpoint
    :param files:
    """
    button_ok = st.button("Predict")
    if button_ok:
        data_ = data[:15]
        output = requests.post(endpoint, files=files, timeout=8000)
        data_["predict"] = output.json()["prediction"]
        st.write(data_.head())
