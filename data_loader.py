"""Модуль для загрузки данных."""

import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Загружает данные из Excel файла.

    Args:
        file_path (str): Путь к файлу данных.

    Returns:
        pd.DataFrame: Загруженный DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл данных не найден: {file_path}")

    print(f"Загрузка данных из {file_path}...")
    df = pd.read_excel(file_path)
    print(f"Данные загружены. Размер: {df.shape}")
    return df

def inspect_data(df: pd.DataFrame) -> None:
    """
    Выводит базовую информацию о данных.

    Args:
        df (pd.DataFrame): DataFrame для инспекции.
    """
    print("\nНазвания столбцов в данных:")
    print(df.columns.tolist())
    print(f"\nКоличество полных дубликатов строк: {df.duplicated().sum()}")
    print("\nПропущенные значения в исходных данных:")
    print(df.isnull().sum())
