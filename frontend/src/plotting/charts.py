"""
Графики EDA
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_text(ax, rotation=0, xytext=(0, 20), fontsize=14):
    """
    функция для расположения процентных значений на графике
    :param ax: график
    :param rotation: угол поворота подписей
    :param xytext: расположение подписей
    :param fontsize: размер шрифта
    :return:
    """
    for p in ax.patches:
        percentage = "{:.1f}%".format(p.get_height())
        ax.annotate(
            percentage,
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=xytext,
            textcoords="offset points",
            rotation=rotation,
            fontsize=fontsize,
        )


def plot_text_cnt(ax):
    """
    функция для расположения числовых значений на графике
    """
    for p in ax.patches:
        percentage = "{:.1f}".format(p.get_height())
        ax.annotate(
            percentage,
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=14,
        )


def age_plot(data: pd.DataFrame):
    """
    Распределение категорий внутри возраста
    :param data: датасет
    :return: график
    """
    fig = plt.figure(figsize=(20, 10))

    age_d = (
        data.groupby(["age_group"])
        .cost_category.value_counts(normalize=True)
        .mul(100)
        .rename("N")
        .reset_index()
    )
    w = sns.barplot(x="age_group", y="N", data=age_d, hue="cost_category",)
    plot_text(w, 45, (0, 22), 14)
    plt.title("Распределение категорий внутри возраста", fontsize=20)
    plt.ylabel("Процент", fontsize=1)
    plt.xlabel("Возрастная группа", fontsize=18)
    plt.legend(loc="upper left", fontsize=15)
    plt.ylim(top=70)
    return fig


def option_plot(data: pd.DataFrame):
    """
    Количество опций в туре по ценовым категориям
    :param data: датасет
    :return: график
    """
    fig = plt.figure(figsize=(15, 10))
    data_tour = (
        data.groupby(["cost_category"])["tour_options"]
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )
    ax1 = sns.barplot(data=data_tour, y="tour_options", x="cost_category")
    plt.title("Количество опций в туре по ценовым категориям", fontsize=15)
    plt.xlabel("Ценовая категория", fontsize=15)
    plt.ylabel("Медианные значения", fontsize=15)
    plot_text_cnt(ax1)
    return fig


def purpose_plot(data: pd.DataFrame):
    """
    Цели визита туристов по каждой ценовой категории
    :param data: датасет
    :return: график
    """
    fig = plt.figure(figsize=(15, 10))
    norm_data = (
        data.groupby(["cost_category"])
        .purpose.value_counts(normalize=True)
        .rename("Norm")
        .reset_index()
    )
    sns.barplot(y="cost_category", x="Norm", data=norm_data, hue="purpose")
    plt.title("Цели визита туристов по каждой ценовой категории", fontsize=20)
    plt.ylabel("Ценовая категория", fontsize=15)
    plt.xlabel("Нормализованный ряд", fontsize=15)
    plt.legend(loc="lower right", fontsize=15)
    return fig


def main_act_plot(data: pd.DataFrame):
    '''
    Основной вид деятельности по каждой категории
    :param data: датасет
    :return: график
    '''
    fig = plt.figure(figsize=(15, 10))
    norm_data = data.groupby([
        'cost_category'
    ]).main_activity.value_counts(normalize=True).rename('Norm').reset_index()
    sns.barplot(y='cost_category', x='Norm', data=norm_data, hue='main_activity')
    plt.title('Основные занятия туристов по каждой ценовой категории', fontsize=20)
    plt.ylabel('Ценовая категория', fontsize=15)
    plt.xlabel('Нормализованный ряд', fontsize=15)
    plt.legend(loc="lower right", fontsize=15)
    return fig


def nights_category(data: pd.DataFrame):
    '''
    Как изменяется кол-во групп от проведенного времени в поездке
    :param data: датасет
    :return: график
    '''
    fig = plt.figure(figsize=(15, 10))
    data['total_nights_bin'] = pd.CategoricalIndex(data['total_nights_bin'],
                        categories=['less_7_days', '8_to_14_days', '15_to_30_days',
                                    '31_to_180_days', 'more_180_days'],
                        ordered=True, dtype='category')
    for category in data.cost_category.unique():
        data_3 = data[data['cost_category'] == category].groupby(
            ['cost_category', 'total_nights_bin']).count().reset_index()
        sns.lineplot(x='total_nights_bin', y='Tour_ID', data=data_3, linewidth=3)
        plt.legend(data.cost_category.unique())
    plt.legend(data.cost_category.unique(), fontsize=15)
    plt.ylabel('Количество групп', fontsize=15)
    plt.xlabel('Количество дней', fontsize=15)
    plt.title('Динамика количества групп по категориям от проведенных ночей', fontsize=20)
    return fig


def cont_category(data: pd.DataFrame, name_continent: str):
    '''
    :param data: датасет
    :param name_continent: название континента, относительно которого хочу посмотреть распределение категорий
    :return: график
    '''
    fig = plt.figure(figsize=(8, 5))
    n_cont = {
        'Europe': 'EU',
        'Africa': "AF",
        'Asia': 'AS',
        'South America': 'South_Am',
        'Other': "other",
        'North America': 'North_Am',
    }
    if name_continent != 'All':
        data_1 = pd.DataFrame(data[data['continent'] == n_cont[name_continent]].groupby(
            ['continent'])['cost_category'].value_counts(
                normalize=True).sort_index().mul(100)).rename(
                    columns={
                        'cost_category': 'Percent'
                    }).reset_index().drop(['continent'], axis=1)

        ax = sns.barplot('cost_category', 'Percent', data=data_1)
        plt.title(name_continent, fontsize=20)
        plt.xlabel('Категория', fontsize=15)
        plt.ylabel('Проценты', fontsize=15)
        plt.ylim(top=60)
        plt.xticks(rotation=45)
        plot_text(ax, 45, (0, 25))


    elif name_continent == 'All':
        fig = plt.figure(figsize=(20, 40))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        list_continents = {
            'EU': 'Европа',
            'North_Am': 'Северная Америка',
            'AF': "Африка",
            'other': "Другие страны",
            'AS': "Азия",
            'South_Am': "Южная Америка"
        }
        len_continents = len(list_continents)
        for cnt, i in enumerate(list_continents.keys()):
            cnt += 1
            plt.subplot(len_continents, 3, cnt)
            data_1 = pd.DataFrame(data[data['continent'] == i].groupby(
                ['continent'])['cost_category'].value_counts(
                normalize=True).sort_index().mul(100)).rename(
                columns={
                    'cost_category': 'Percent'
                }).reset_index().drop(['continent'], axis=1)
            ax = sns.barplot('cost_category', 'Percent', data=data_1)
            plt.title(list_continents[i], fontsize=20)
            plt.xlabel('Категория', fontsize=15)
            plt.ylabel('Проценты', fontsize=15)
            plt.ylim(top=60)
            plt.xticks(rotation=45)
            plot_text(ax, 45, (0, 25))
    return fig