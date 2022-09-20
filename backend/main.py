"""
Модель для помощи в оценивании расходов туристической группы и отнесение ее к определенной ценовой категории
"""
import pandas as pd
import warnings
import optuna
import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel
from src.pipelinies.pipeline_train import pipeline_training
from src.pipelinies.pipeline_evaluate import *
from src.train.metrics import load_metrics
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = FastAPI()
CONFIG_PATH = "../config/parameters.yml"


class CostCategory(BaseModel):
    """
    признаки для определения ценовой категории туристической группы
    """

    country: str
    age_group: str
    travel_with: str
    total_female: int
    total_male: int
    purpose: str
    main_activity: str
    info_source: str
    tour_arrangement: str
    package_transport_int: str
    package_accomodation: str
    package_food: str
    package_transport_tz: str
    package_sightseeing: str
    package_guided_tour: str
    package_insurance: str
    night_mainland: int
    night_zanzibar: int
    first_trip_tz: str


@app.post("/train")
def training():
    """
    обучение модели, подгрузка config
    """
    pipeline_training(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH)
    return {"metrics": metrics}


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    """
    получение предсказаний из файла
    """
    result = pipeline_evaluating(config_path=CONFIG_PATH, data_path=file.file)
    return {"prediction": result[:15]}


@app.post("/predict_input")
def prediction_input_features(tour_group: CostCategory):
    """
    предсказание ценовой категории по вводимым данным
    """
    features = [
        [
            tour_group.country,
            tour_group.age_group,
            tour_group.travel_with,
            tour_group.total_female,
            tour_group.total_male,
            tour_group.purpose,
            tour_group.main_activity,
            tour_group.info_source,
            tour_group.tour_arrangement,
            tour_group.package_transport_int,
            tour_group.package_accomodation,
            tour_group.package_food,
            tour_group.package_transport_tz,
            tour_group.package_sightseeing,
            tour_group.package_guided_tour,
            tour_group.package_insurance,
            tour_group.night_mainland,
            tour_group.night_zanzibar,
            tour_group.first_trip_tz,
        ]
    ]

    columns = [
        "country",
        "age_group",
        "travel_with",
        "total_female",
        "total_male",
        "purpose",
        "main_activity",
        "info_source",
        "tour_arrangement",
        "package_transport_int",
        "package_accomodation",
        "package_food",
        "package_transport_tz",
        "package_sightseeing",
        "package_guided_tour",
        "package_insurance",
        "night_mainland",
        "night_zanzibar",
        "first_trip_tz",
    ]

    df = pd.DataFrame(features, columns=columns)
    predictions = pipeline_evaluating(config_path=CONFIG_PATH, data=df)[0]
    dict_values = {"Normal Cost": "The tourist group category is Normal cost",
                   "Higher Cost": "The tourist group category is Higher cost",
                   "High Cost": "The tourist group category is High cost",
                   "Highest Cost": "The tourist group category is Highest cost",
                   "Low Cost": "The tourist group category is Low cost",
                   "Lower Cost": "The tourist group category is Lower cost"}
    result = dict_values.get(predictions)
    return result


if __name__ == "__main__":
    # Запуск
    uvicorn.run(app, host="127.0.0.1", port=80)