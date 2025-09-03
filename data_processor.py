"""Модуль для предобработки и создания новых признаков."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает данные и подготавливает их для анализа.

    Args:
        df (pd.DataFrame): Исходный DataFrame.

    Returns:
        pd.DataFrame: Очищенный и подготовленный DataFrame.
    """
    print("Очистка и подготовка данных...")
    df_copy = df.copy()
    
    # Преобразование столбца времени
    df_copy['Время'] = pd.to_datetime(df_copy['Время'])
    df_copy = df_copy.sort_values('Время').reset_index(drop=True)

    # Переименование столбца, если необходимо
    if 'Рабочий парк.1' in df_copy.columns:
        df_copy = df_copy.rename(columns={'Рабочий парк.1': 'Рабочий парк четный'})
        print("Столбец 'Рабочий парк.1' переименован в 'Рабочий парк четный'")

    # Удаление строк с критическими пропусками
    critical_columns = ['Время', 'Простой факт', 'Рабочий парк', 'Рабочий парк нечетный', 'Рабочий парк четный']
    df_clean = df_copy.dropna(subset=critical_columns)
    print(f"Исходный размер данных: {df_copy.shape[0]}")
    print(f"Размер данных после очистки: {df_clean.shape[0]}")
    
    return df_clean

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создает дополнительные признаки.

    Args:
        df (pd.DataFrame): DataFrame с основными данными.

    Returns:
        pd.DataFrame: DataFrame с новыми признаками.
    """
    print("Создание дополнительных признаков...")
    df_features = df.copy()
    
    # Создание дополнительных признаков
    df_features['дисбаланс_направлений'] = abs(df_features['Рабочий парк нечетный'] - df_features['Рабочий парк четный'])
    df_features['переработка'] = np.where(df_features['Рабочий парк'] != 0,
                                          df_features['Отправление общее'] / df_features['Рабочий парк'], 0)
    df_features['отношение_прибытие_отправление'] = np.where(df_features['Отправление общее'] != 0,
                                                            df_features['Прибытие общее'] / df_features['Отправление общее'], 0)

    # Добавление временных признаков
    df_features['Час'] = df_features['Время'].dt.hour
    df_features['День_недели'] = df_features['Время'].dt.dayofweek  # 0 - понедельник, 6 - воскресенье
    df_features['Месяц'] = df_features['Время'].dt.month
    
    print("Признаки созданы.")
    print("\nДанные после преобразований (первые 5 строк):")
    print(df_features[['Время', 'Рабочий парк', 'Рабочий парк нечетный', 'Рабочий парк четный',
                    'дисбаланс_направлений', 'переработка', 'отношение_прибытие_отправление']].head())
    return df_features

def create_lagged_features(df: pd.DataFrame, target_cols: list, lags: list = [1, 2, 3]) -> pd.DataFrame:
    """
    Создает лаговые признаки для указанных столбцов.

    Args:
        df (pd.DataFrame): Исходный DataFrame.
        target_cols (list): Список столбцов для создания лагов.
        lags (list): Список лагов.

    Returns:
        pd.DataFrame: DataFrame с лаговыми признаками.
    """
    print("Создание лаговых признаков...")
    df_lagged = df.copy()
    for col in target_cols:
        for lag in lags:
            df_lagged[f'{col}_lag_{lag}'] = df_lagged[col].shift(lag)
    # Удаляем строки с пропущенными значениями после создания лагов
    df_lagged = df_lagged.dropna()
    print(f"Лаговые признаки созданы. Новый размер: {df_lagged.shape}")
    return df_lagged