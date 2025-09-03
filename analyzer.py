"""Модуль для анализа данных и кластеризации."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Оптимизированная реализация субтрактивной кластеризации
def subtractive_clustering(data, ra, rb=None, max_iterations=50):
    """
    Оптимизированная реализация алгоритма субтрактивной кластеризации.
    """
    print(f"Выполнение субтрактивной кластеризации с ra={ra}...")
    if rb is None:
        rb = 1.5 * ra
    # Нормализация данных
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    n = len(data_scaled)
    # === ОПТИМИЗАЦИЯ 1: Используем матричные операции для вычисления плотности ===
    # Создаем матрицу квадратов расстояний
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        dist_matrix[i] = np.sum((data_scaled - data_scaled[i])**2, axis=1)
    # Вычисляем плотность сразу для всех точек
    density = np.sum(np.exp(-dist_matrix / (ra**2)), axis=1)
    # === ОПТИМИЗАЦИЯ 2: Добавляем защиту от бесконечного цикла ===
    centers = []
    iterations = 0
    max_density_initial = np.max(density)
    min_density_threshold = 1e-10  # Абсолютный порог для завершения
    while iterations < max_iterations:
        # Найти точку с максимальной плотностью
        idx = np.argmax(density)
        current_max_density = density[idx]
        # Проверка условий завершения
        if current_max_density < 0.15 * max_density_initial or \
           current_max_density < min_density_threshold or \
           np.isnan(current_max_density):
            break
        centers.append(idx)
        # === ОПТИМИЗАЦИЯ 3: Векторизованное обновление плотности ===
        # Используем предварительно вычисленную матрицу расстояний
        density = density - current_max_density * np.exp(-dist_matrix[:, idx] / (rb**2))
        iterations += 1
    # Назначение кластеров
    clusters = np.zeros(n)
    if centers:
        # Создаем матрицу расстояний до центров
        center_matrix = dist_matrix[:, centers]
        clusters = np.argmin(center_matrix, axis=1)
    print(f"Кластеризация завершена. Найдено {len(centers)} кластеров.")
    return np.array(centers), clusters

def find_optimal_ra(X_cluster: np.ndarray, ra_values: list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]) -> float:
    """
    Подбирает оптимальный параметр ra для субтрактивной кластеризации.
    """
    print("Подбор оптимального параметра ra для субтрактивной кластеризации...")
    cluster_counts = []
    for ra in ra_values:
        centers, clusters = subtractive_clustering(X_cluster, ra)
        cluster_counts.append(len(centers))
        print(f"ra = {ra}, количество кластеров: {len(centers)}")

    # Определение оптимального ra (точка "перелома" в количестве кластеров)
    optimal_ra = ra_values[0]
    if len(cluster_counts) > 1:
        diffs = np.diff(cluster_counts)
        if len(diffs) > 0:
            optimal_idx = np.argmax(diffs) + 1
            optimal_ra = ra_values[optimal_idx]
    print(f"Оптимальный параметр ra: {optimal_ra}")
    return optimal_ra

def perform_clustering(df_sample: pd.DataFrame, df_clean: pd.DataFrame, clustering_features: list, MAX_SAMPLE_SIZE: int) -> pd.DataFrame:
    """
    Выполняет кластеризацию и добавляет результаты в DataFrame.
    """
    X_cluster = df_sample[clustering_features].values

    # Подбор оптимального ra
    optimal_ra = find_optimal_ra(X_cluster)

    # Проведение кластеризации с оптимальным ra
    centers_idx, clusters = subtractive_clustering(X_cluster, optimal_ra)

    # Добавляем кластеры в исходный DataFrame
    df_result = df_clean.copy()
    if len(df_clean) > MAX_SAMPLE_SIZE:
        # Назначаем кластеры всем точкам на основе подвыборки
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_cluster, clusters)
        df_result['Кластер'] = knn.predict(df_result[clustering_features].values)
    else:
        df_result['Кластер'] = clusters

    # Анализ кластеров
    cluster_analysis = df_result.groupby('Кластер').mean()[clustering_features]
    print("\nХарактеристики кластеров:")
    print(cluster_analysis)
    
    return df_result, cluster_analysis, centers_idx, clusters, optimal_ra

def analyze_anomalies(df_clean: pd.DataFrame, anomaly_threshold_dt: float = 15.0, anomaly_threshold_park: float = 5000.0) -> pd.DataFrame:
    """
    Анализирует аномалии в данных.
    """
    print("Анализ аномалий...")
    df_result = df_clean.copy()
    # Создаем столбцы для аномалий
    df_result['Аномальный_простой'] = (df_result['Простой факт'] > anomaly_threshold_dt).astype(int)
    df_result['Аномальный_парк'] = (df_result['Рабочий парк'] > anomaly_threshold_park).astype(int)
    
    print("\nАнализ аномалий по кластерам:")
    for cluster in np.unique(df_result['Кластер']):
        cluster_data = df_result[df_result['Кластер'] == cluster]
        anomaly_dt = np.sum(cluster_data['Простой факт'] > anomaly_threshold_dt)
        anomaly_park = np.sum(cluster_data['Рабочий парк'] > anomaly_threshold_park)
        total = len(cluster_data)
        print(f"Кластер {cluster}:")
        print(f"  - Аномалии простоя (>15 ч): {anomaly_dt}/{total} ({anomaly_dt/total:.1%})")
        print(f"  - Аномалии рабочего парка (>5000): {anomaly_park}/{total} ({anomaly_park/total:.1%})")
        print(f"  - Средний простой: {cluster_data['Простой факт'].mean():.2f} ч")
        print(f"  - Средний рабочий парк: {cluster_data['Рабочий парк'].mean():.2f} ваг.")
        
    return df_result