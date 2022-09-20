"""
Frontend
"""

import os
import yaml
import joblib
import streamlit as st
from src.get_data.get_data import load_data, get_dataset
from src.plotting.charts import *
from src.train.training import start_training
from src.evaluate.evaluate import evaluate_input, evaluate_from_file

CONFIG_PATH = "../config/parameters.yml"


def main_page():
    """
    Страница с описанием проекта
    """
    st.image(
        "https://world-card.ru/images/Priroda/Tanzania4.jpg", width=700,
    )

    st.markdown("# Описание проекта ")
    st.title("MLOps project: AI4D Africa’s Anglophone Research Lab Tanzania Tourism Classification Challenge ")
    st.write(
        """
        Набор данных описывает актуальную информацию о туристических расходах, собранных 
        Национальным бюро статистики (НБС) Танзании. Набор данных был собран, чтобы лучше понять состояние сектора 
        туризма и предоставить инструмент, который будет обеспечить рост отрасли.
        Необходимо классифицировать диапазон расходов, которые турист тратит в Танзании. 
        Модель может использоваться различными туроператорами и Советом по туризму Танзании, 
        чтобы автоматически помочь туристам со всего мира оценить свои расходы перед посещением Танзании. 
        
        Target: классификация расходов туристических групп предполагает 6 категорий: Highest Cost, Higher Cost, High Cost,
        Normal Cost, Low Cost, Lower Cost. Стоит обратить внимание, что есть дисбаланс классов в данных.
        
        Показателем оценки для этой задачи является logloss
        """
    )

    # name of the columns
    st.markdown(
        """
        ### Описание полей 
            -Tour_ID - Уникальный идентификатор для каждого туриста
            -country - Страна, из которой приехал турист
            -age_group - Возрастная группа туриста
            -travel_with - Отношение людей, с которыми путешествуют туристы по Танзании
            -total_female - Общее количество женщин в тур.группе
            -total_male - Общее количество мужчин в тур.группе
            -purpose - Цель посещения Танзании
            -main_activity - Основное направление туризма в Танзании
            -info_source - Источник информации о туризме в Танзании
            -tour_arrangment - Организация посещения Танзании
            -package_transport_int - Если турпакет включает международные перевозки
            -package_accomodation - Если в турпакет входит услуга проживания
            -package_food - Если в турпакет включено питание
            -package_transport_tz - Если турпакет включает транспортное обслуживание в пределах Танзании
            -package_sightseeing - Если турпакет включает экскурсионное обслуживание
            -package_guided_tour - Если турпакет включает гида
            -package_insurance - если турпакет включает страховку
            -night_mainland - Количество ночей, проведенных туристом на материковой части Танзании
            -night_zanzibar - Количество ночей, проведенных туристом на Занзибаре *Занзибар - остров
            -first_trip_tz - Если бы это была первая поездка в Танзанию
            -cost_category - Диапазон расходов, которые турист тратит в Танзании
    """
    )


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown("# Exploratory data analysis️")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # load and write dataset
    data = get_dataset(dataset_path=config["preprocessing"]["data_for_plot"])
    st.write(data.head())
    st.markdown('Для EDA были созданы признаки: total_person - всего человек в группе, tour_options - сумма тур.опций, '
                'total_nights - всего проведенных ночей в стране, total_nights_bin - разделние total_nights на бины,'
                'continent - континент' )

    # plotting with checkbox
    age_group = st.sidebar.checkbox("Распределение категорий внутри возраста")
    option_tour = st.sidebar.checkbox("Количество опций в туре по ценовым категориям")
    purpose_group = st.sidebar.checkbox(
        "Цели визита туристов по каждой ценовой категории"
    )
    nigts_group = st.sidebar.checkbox('Количество групп - количество дней')
    activ_group = st.sidebar.checkbox('Категорий - основные занятия в поездке')
    continent_group = st.sidebar.selectbox('Континент - категории', [None, 'Europe', 'Africa', 'Asia','North America',
                                                                     'South America', 'Other', 'All'])

    if age_group:
        st.pyplot(age_plot(data=data))
        # выводы
        st.markdown('**Вывод:** из графика следует, что более возрастнные группы (45+) оставляют больше денег, '
                    'чем остальные. Гипотеза подтвердилась.')
        st.markdown('Также видно, '
                    'что возрастная категория <18 тоже имеет большую долю высокой ценовой категории. '
                    'Возможно,что это связано с тем, что детям оплачивают поездку родители, тогда как категории 18-24 '
                    'и 25-44 вынуждены сами оплачиваеть поездку.')

    if option_tour:
        st.pyplot(option_plot(data=data))
        # выводы
        st.markdown('**Вывод:** чем больше опций включено в тур (путевку), тем больше денег принест '
                    'туристическая группа')

    if purpose_group:
        st.pyplot(purpose_plot(data=data))
        # выводы
        st.markdown('**Вывод:** туристические группы, которые относятся к Highest, Higher, High cost, '
                    'в основном приезжают для проведения отпуска. Значит, если цель группы отдых, то скорее всего '
                    'она потратит больше денег, чем в случае иной цели визита.')

    if nigts_group:
        st.pyplot(nights_category(data=data))
        # выводы
        st.markdown('**Вывод:** из графика видно, что при поездке  от 8 до 14 дней, количесвто групп из категорий '
                    'Highest, Higher  и High увеличивается. Ранее было установлено, что данные категории в основном'
                    ' приезжают для проведения отпуска. Можно сделать вывод, что категории Highest, Higher  и High '
                    'приезжают на одну-две недели в отпуск')

    if activ_group:
        st.pyplot(main_act_plot(data=data))
        # выводы
        st.markdown('**Вывод:** WildLife tourism является наиболее распространненной целью визита, и количество '
                    'категорий High, Higher, Highest преобладает над другими категориями в разрезе основного занятия '
                    'тур.группы ')

    if continent_group:
        st.pyplot(cont_category(data=data, name_continent=continent_group))
        st.markdown('**Вывод:** из стран Европы, Северной и Южной Америк в основном едут '
                    'обеспеченные туристы, которые оставляют много денег')
        st.markdown('Из Африки в основном приезжают туристы, которые оставляют мало денег')
        st.markdown('Возможно, это следует из-за уровня развития региона')


def training():
    """
    Тренировка модели
    """
    st.markdown("# Training model - LightGBM")

    st.caption("Click to start training the model")
    # get params
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # endpoint
    endpoint = config["endpoints"]["train"]

    if st.button("Start training"):
        start_training(config=config, endpoint=endpoint)


def prediction():
    """
    Получение предсказаний путем ввода данных
    """
    st.markdown("# Prediction")
    st.image(
        "https://process.filestackapi.com/resize=fit:clip,width:1440,height:1000/quality=v:79/compress/cache=expiry:604800/d1wgSE7SP69qp00gnxbT",
        width=700,
    )
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_input"]
    unique_data_path = config["preprocessing"]["unique_values_path"]

    # проверка на наличие сохраненной модели
    if os.path.exists(config["train"]["model_path"]):
        evaluate_input(unique_data_path=unique_data_path, endpoint=endpoint)
    else:
        st.error("Сначала обучите модель")


def prediction_from_file():
    """
    Получение предсказаний из файла с данными
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_from_file"]

    upload_file = st.file_uploader(
        "", type=["csv", "xlsx"], accept_multiple_files=False
    )
    # проверка загружен ли файл
    if upload_file:
        dataset_csv_df, files = load_data(data=upload_file, type_data="Test")
        # проверка на наличие сохраненной модели
        if os.path.exists(config["train"]["model_path"]):
            evaluate_from_file(data=dataset_csv_df, endpoint=endpoint, files=files)
        else:
            st.error("Сначала обучите модель")


def main():
    """
    Сборка пайплайна в одном блоке
    """
    page_names_to_funcs = {
        "Описание проекта": main_page,
        "Exploratory data analysis": exploratory,
        "Training model": training,
        "Prediction": prediction,
        "Prediction from file": prediction_from_file,
    }
    selected_page = st.sidebar.selectbox("Выберите пункт", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
