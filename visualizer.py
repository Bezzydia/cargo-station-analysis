"""Модуль для визуализации данных."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Установка стиля
sns.set_style("whitegrid")
sns.set_palette("husl")

def save_and_show_plot(filename: str, results_dir: str = "results/figures") -> None:
    """Сохраняет текущий график и показывает его."""
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close() # Важно закрывать, чтобы не накапливались фигуры

def plot_time_series(df: pd.DataFrame) -> None:
    """Строит графики временных рядов."""
    print("Построение графиков временных рядов...")
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    axes[0, 0].plot(df['Время'], df['Простой факт'])
    axes[0, 0].set_title('Простой факт')
    axes[0, 0].set_ylabel('Часы')

    axes[0, 1].plot(df['Время'], df['Рабочий парк'])
    axes[0, 1].set_title('Рабочий парк')
    axes[0, 1].set_ylabel('Вагоны')

    axes[1, 0].plot(df['Время'], df['t(1) - Прибытие'], label='Прибытие')
    axes[1, 0].plot(df['Время'], df['t(c) - Сортировка'], label='Сортировка')
    axes[1, 0].plot(df['Время'], df['t(o) - Отправление'], label='Отправление')
    axes[1, 0].set_title('Временные операции')
    axes[1, 0].set_ylabel('Часы')
    axes[1, 0].legend()

    axes[1, 1].plot(df['Время'], df['Прибытие общее'], label='Прибытие')
    axes[1, 1].plot(df['Время'], df['Отправление общее'], label='Отправление')
    axes[1, 1].set_title('Интенсивность прибытия и отправления')
    axes[1, 1].set_ylabel('Вагоны/час')
    axes[1, 1].legend()

    axes[2, 0].plot(df['Время'], df['дисбаланс_направлений'])
    axes[2, 0].set_title('Дисбаланс направлений')
    axes[2, 0].set_ylabel('Вагоны')

    axes[2, 1].plot(df['Время'], df['Рабочий парк нечетный'], label='Нечетный')
    axes[2, 1].plot(df['Время'], df['Рабочий парк четный'], label='Четный')
    axes[2, 1].set_title('Рабочий парк по направлениям')
    axes[2, 1].set_ylabel('Вагоны')
    axes[2, 1].legend()

    save_and_show_plot('eda_time_series.png')
    print("Графики временных рядов сохранены.")

def plot_distributions(df: pd.DataFrame) -> None:
    """Строит гистограммы распределений."""
    print("Построение гистограмм распределений...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    sns.histplot(df['Простой факт'], ax=axes[0, 0], kde=True)
    axes[0, 0].set_title('Распределение простоя')

    sns.histplot(df['Рабочий парк'], ax=axes[0, 1], kde=True)
    axes[0, 1].set_title('Распределение рабочего парка')

    sns.histplot(df['дисбаланс_направлений'], ax=axes[1, 0], kde=True)
    axes[1, 0].set_title('Распределение дисбаланса направлений')

    sns.histplot(df['переработка'], ax=axes[1, 1], kde=True)
    axes[1, 1].set_title('Распределение интенсивности переработки')

    save_and_show_plot('eda_distributions.png')
    print("Гистограммы распределений сохранены.")

def plot_correlation_matrix(df: pd.DataFrame, numeric_columns: list) -> None:
    """Строит тепловую карту корреляционной матрицы."""
    print("Построение корреляционной матрицы...")
    plt.figure(figsize=(16, 14))
    correlation_matrix = df[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                cbar_kws={'shrink': 0.8}, annot_kws={"size": 8})
    plt.title('Матрица корреляций', fontsize=16)
    save_and_show_plot('correlation_matrix.png')
    print("Корреляционная матрица сохранена.")
    return correlation_matrix

def plot_time_dependencies(df_lagged: pd.DataFrame) -> None:
    """Строит графики временных зависимостей (лаги)."""
    print("Построение графиков временных зависимостей...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Корреляция простоя с лагами рабочего парка
    for lag in [1, 2, 3]:
        sns.scatterplot(x=f'Рабочий парк_lag_{lag}', y='Простой факт',
                        data=df_lagged, ax=axes[0, 0], alpha=0.5, label=f'Лаг {lag}')
    axes[0, 0].set_title('Зависимость простоя от рабочего парка в прошлом')
    axes[0, 0].set_xlabel('Рабочий парк (лаг)')
    axes[0, 0].set_ylabel('Простой факт')
    axes[0, 0].legend()

    # Корреляция рабочего парка с лагами прибытия
    for lag in [1, 2, 3]:
        sns.scatterplot(x=f'Прибытие общее_lag_{lag}', y='Рабочий парк',
                        data=df_lagged, ax=axes[0, 1], alpha=0.5, label=f'Лаг {lag}')
    axes[0, 1].set_title('Зависимость рабочего парка от прибытия в прошлом')
    axes[0, 1].set_xlabel('Прибытие общее (лаг)')
    axes[0, 1].set_ylabel('Рабочий парк')
    axes[0, 1].legend()

    # Корреляция простоя с лагами сортировки
    for lag in [1, 2, 3]:
        sns.scatterplot(x=f't(c) - Сортировка_lag_{lag}', y='Простой факт',
                        data=df_lagged, ax=axes[1, 0], alpha=0.5, label=f'Лаг {lag}')
    axes[1, 0].set_title('Зависимость простоя от сортировки в прошлом')
    axes[1, 0].set_xlabel('t(c) - Сортировка (лаг)')
    axes[1, 0].set_ylabel('Простой факт')
    axes[1, 0].legend()

    # Корреляция рабочего парка с лагами простоя
    for lag in [1, 2, 3]:
        sns.scatterplot(x=f'Простой факт_lag_{lag}', y='Рабочий парк',
                        data=df_lagged, ax=axes[1, 1], alpha=0.5, label=f'Лаг {lag}')
    axes[1, 1].set_title('Зависимость рабочего парка от простоя в прошлом')
    axes[1, 1].set_xlabel('Простой факт (лаг)')
    axes[1, 1].set_ylabel('Рабочий парк')
    axes[1, 1].legend()

    save_and_show_plot('time_dependencies.png')
    print("Графики временных зависимостей сохранены.")

def plot_feature_importance(feature_importance_series, title: str, filename: str) -> None:
    """Строит график важности признаков."""
    print(f"Построение графика важности признаков: {title}...")
    plt.figure(figsize=(10, 6))
    feature_importance_series.sort_values().plot(kind='barh')
    plt.title(title)
    plt.xlabel('Важность')
    save_and_show_plot(filename)
    print(f"График важности признаков сохранен как {filename}.")

def plot_clustering_results(df_clean: pd.DataFrame) -> None:
    """Строит графики результатов кластеризации."""
    print("Построение графиков кластеризации...")
    plt.figure(figsize=(14, 12))

    # Кластеры в пространстве "Простой факт" vs "Рабочий парк"
    plt.subplot(2, 2, 1)
    for cluster in np.unique(df_clean['Кластер']):
        cluster_data = df_clean[df_clean['Кластер'] == cluster]
        plt.scatter(
            cluster_data['Рабочий парк'],
            cluster_data['Простой факт'],
            alpha=0.6, label=f'Кластер {int(cluster)}', s=10
        )
    plt.xlabel('Рабочий парк')
    plt.ylabel('Простой факт')
    plt.title('Рабочий парк vs Простой факт по кластерам')
    plt.legend()

    # Кластеры в пространстве "t(c) - Сортировка" vs "Простой факт"
    plt.subplot(2, 2, 2)
    for cluster in np.unique(df_clean['Кластер']):
        cluster_data = df_clean[df_clean['Кластер'] == cluster]
        plt.scatter(
            cluster_data['t(c) - Сортировка'],
            cluster_data['Простой факт'],
            alpha=0.6, label=f'Кластер {int(cluster)}', s=10
        )
    plt.xlabel('t(c) - Сортировка')
    plt.ylabel('Простой факт')
    plt.title('Сортировка vs Простой факт по кластерам')
    plt.legend()

    # Распределение кластеров и аномалий
    plt.subplot(2, 2, 3)
    anomaly_rates_dt = df_clean.groupby('Кластер')['Аномальный_простой'].mean()
    anomaly_rates_park = df_clean.groupby('Кластер')['Аномальный_парк'].mean()
    x = np.arange(len(anomaly_rates_dt))
    width = 0.35
    plt.bar(x - width/2, anomaly_rates_dt, width, label='Простой > 15 ч')
    plt.bar(x + width/2, anomaly_rates_park, width, label='Парк > 5000 ваг.')
    plt.xlabel('Кластер')
    plt.ylabel('Доля аномалий')
    plt.title('Распределение аномалий по кластерам')
    plt.xticks(x, [f'{i}' for i in anomaly_rates_dt.index])
    plt.legend()

    save_and_show_plot('subtractive_clustering.png')
    print("Графики кластеризации сохранены.")

def plot_lstm_results(history, y_test_inv, test_predict_inv) -> None:
    """Строит графики результатов LSTM."""
    print("Построение графиков LSTM...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Обучение')
    plt.plot(history.history['val_loss'], label='Валидация')
    plt.title('Функция потерь LSTM')
    plt.xlabel('Эпохи')
    plt.ylabel('MSE')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(y_test_inv, label='Фактические значения', alpha=0.7)
    plt.plot(test_predict_inv, label='Прогноз LSTM', alpha=0.7)
    plt.title('Прогнозирование простоя с использованием LSTM')
    plt.xlabel('Временные шаги')
    plt.ylabel('Простой (часы)')
    plt.legend()

    save_and_show_plot('lstm_results.png')
    print("Графики LSTM сохранены.")

def plot_anomaly_temporal_analysis(df_clean: pd.DataFrame) -> None:
    """Строит графики временных паттернов аномалий."""
    print("Построение графиков временных паттернов аномалий...")
    # Создаем временные признаки для анализа (если еще не созданы)
    if 'Час' not in df_clean.columns:
         df_clean['Час'] = df_clean['Время'].dt.hour
    if 'День_недели' not in df_clean.columns:
        df_clean['День_недели'] = df_clean['Время'].dt.dayofweek
    if 'Месяц' not in df_clean.columns:
        df_clean['Месяц'] = df_clean['Время'].dt.month

    anomaly_by_hour = df_clean.groupby('Час')['Аномальный_простой'].mean()
    anomaly_by_day = df_clean.groupby('День_недели')['Аномальный_простой'].mean()
    anomaly_by_month = df_clean.groupby('Месяц')['Аномальный_простой'].mean()

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(anomaly_by_hour.index, anomaly_by_hour.values, 'o-')
    plt.title('Доля аномалий простоя по часам суток')
    plt.xlabel('Час')
    plt.ylabel('Доля аномалий')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(anomaly_by_day.index, anomaly_by_day.values, 'o-')
    plt.title('Доля аномалий простоя по дням недели')
    plt.xlabel('День недели (0-пн, 6-вс)')
    plt.ylabel('Доля аномалий')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(anomaly_by_month.index, anomaly_by_month.values, 'o-')
    plt.title('Доля аномалий простоя по месяцам')
    plt.xlabel('Месяц')
    plt.ylabel('Доля аномалий')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    anomaly_rates_by_cluster = df_clean.groupby('Кластер')['Аномальный_простой'].mean()
    plt.bar(anomaly_rates_by_cluster.index, anomaly_rates_by_cluster.values)
    plt.title('Доля аномалий простоя по кластерам')
    plt.xlabel('Кластер')
    plt.ylabel('Доля аномалий')
    plt.grid(True, alpha=0.3)

    save_and_show_plot('anomaly_temporal_analysis.png')
    print("Графики временных паттернов аномалий сохранены.")