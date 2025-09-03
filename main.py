"""Точка входа в приложение."""

import numpy as np
import tensorflow as tf
# Установка seed для воспроизводимости
np.random.seed(42)
tf.random.set_seed(42)

# Импорт модулей
from src.data_loader import load_data, inspect_data
from src.data_processor import clean_and_prepare_data, create_features, create_lagged_features
from src.visualizer import (
    plot_time_series, plot_distributions, plot_correlation_matrix,
    plot_time_dependencies, plot_clustering_results, plot_feature_importance,
    plot_lstm_results, plot_anomaly_temporal_analysis
)
from src.analyzer import perform_clustering, analyze_anomalies
from src.modeler import (
    prepare_modeling_data, train_and_evaluate_regression_models,
    time_series_cv, train_lstm_model, train_svm_anomaly_model, save_models_and_results,
    evaluate_model # Импортируем для использования в финальном выводе
)

def main():
    """Главная функция, запускающая весь анализ."""
    print("🚀 НАЧАЛО АНАЛИЗА РАБОТЫ ЖЕЛЕЗНОДОРОЖНОЙ СТАНЦИИ")
    print("=" * 50)

    # --- 1. Загрузка и инспекция данных ---
    file_path = 'data/Входные_данные.xlsx'
    df_raw = load_data(file_path)
    inspect_data(df_raw)

    # --- 2. Предобработка данных ---
    df_clean_step1 = clean_and_prepare_data(df_raw)
    df_clean = create_features(df_clean_step1)

    # --- 3. Визуализация (EDA) ---
    plot_time_series(df_clean)
    plot_distributions(df_clean)

    numeric_columns = [
        't(1) - Прибытие', 't(p) - Расформирование', 't(c) - Сортировка', 't(o) - Отправление',
        'Простой факт', 'Простой без переработки факт',
        'Прибытие общее', 'Отправление общее',
        'Прибытие с запада', 'Отправление на восток',
        'Прибытие с востока', 'Отправление на запад',
        'Рабочий парк', 'Рабочий парк нечетный', 'Рабочий парк четный',
        'дисбаланс_направлений', 'переработка', 'отношение_прибытие_отправление',
        'Час', 'День_недели', 'Месяц'
    ]
    corr_matrix = plot_correlation_matrix(df_clean, numeric_columns)

    # --- 4. Анализ временных зависимостей (лаги) ---
    target_columns = ['Простой факт', 'Рабочий парк', 'Прибытие общее', 'Отправление общее', 't(c) - Сортировка']
    df_lagged = create_lagged_features(df_clean, target_columns, lags=[1, 2, 3])
    # Пересчитываем матрицу корреляций для данных с лагами, если нужно
    # corr_matrix_lagged = df_lagged.corr() 
    plot_time_dependencies(df_lagged)

    # --- 5. Кластеризация ---
    MAX_SAMPLE_SIZE = 2000
    df_sample = df_clean.iloc[np.random.choice(len(df_clean), size=min(MAX_SAMPLE_SIZE, len(df_clean)), replace=False)].copy() if len(df_clean) > MAX_SAMPLE_SIZE else df_clean.copy()

    clustering_features = [
        't(1) - Прибытие', 't(c) - Сортировка', 'Простой факт',
        'Рабочий парк', 'дисбаланс_направлений', 'переработка',
        'Прибытие общее', 'Отправление общее'
    ]
    df_with_clusters, cluster_analysis, centers_idx, clusters, optimal_ra = perform_clustering(
        df_sample, df_clean, clustering_features, MAX_SAMPLE_SIZE
    )
    plot_clustering_results(df_with_clusters)

    # --- 6. Анализ аномалий ---
    df_with_anomalies = analyze_anomalies(df_with_clusters)
    # Добавим аномальные признаки в df_clean для последующего использования
    df_clean['Аномальный_простой'] = df_with_anomalies['Аномальный_простой']
    df_clean['Аномальный_парк'] = df_with_anomalies['Аномальный_парк']

    # --- 7. Подготовка данных для моделирования ---
    (X_train_dt, X_test_dt, y_train_dt, y_test_dt,
     X_train_park, X_test_park, y_train_park, y_test_park) = prepare_modeling_data(df_clean)

    # --- 8. Моделирование (Регрессия) ---
    models_dt, models_park, metrics_dt, metrics_park = train_and_evaluate_regression_models(
        X_train_dt, X_test_dt, y_train_dt, y_test_dt,
        X_train_park, X_test_park, y_train_park, y_test_park
    )

    # --- 9. Кросс-валидация ---
    print("\n=== КРОСС-ВАЛИДАЦИЯ С ВРЕМЕННЫМ РАЗБИЕНИЕМ ===")
    # Предполагаем, что у нас есть скалеры из моделирования
    scaler_dt = models_dt['Ridge'][1] # Получаем scaler из кортежа (model, scaler)
    X_train_dt_scaled = scaler_dt.transform(X_train_dt) if scaler_dt else X_train_dt.values
    scaler_park = models_park['Ridge'][1]
    X_train_park_scaled = scaler_park.transform(X_train_park) if scaler_park else X_train_park.values

    cv_results = {}
    for name, (model, _) in models_dt.items():
        X_for_cv = X_train_dt_scaled if name == 'Ridge' else X_train_dt.values
        cv_results[f'{name}_downtime'] = time_series_cv(model, X_for_cv, y_train_dt, name)

    for name, (model, _) in models_park.items():
        X_for_cv = X_train_park_scaled if name == 'Ridge' else X_train_park.values
        cv_results[f'{name}_park'] = time_series_cv(model, X_for_cv, y_train_park, name)

    # --- 10. Моделирование (LSTM) ---
    lstm_features = [
        't(1) - Прибытие', 't(c) - Сортировка', 'Прибытие общее', 'Отправление общее',
        'Рабочий парк нечетный', 'Рабочий парк четный', 'дисбаланс_направлений', 'Простой факт'
    ]
    lstm_model, history, lstm_preds, lstm_metrics, scaler_lstm = train_lstm_model(df_clean, lstm_features, epochs=10)
    y_test_inv, test_predict_inv = lstm_preds
    mae_lstm, rmse_lstm, r2_lstm = lstm_metrics
    plot_lstm_results(history, y_test_inv, test_predict_inv)

    # --- 11. Моделирование (Классификация аномалий - SVM) ---
    features_initial_dt = [
        't(1) - Прибытие', 't(p) - Расформирование', 't(c) - Сортировка', 't(o) - Отправление',
        'Прибытие общее', 'Отправление общее',
        'Прибытие с запада', 'Отправление на восток',
        'Прибытие с востока', 'Отправление на запад',
        'Рабочий парк нечетный', 'Рабочий парк четный',
        'дисбаланс_направлений'
    ]
    svm_dt, scaler_an_dt, svm_preds, feature_importance_svm = train_svm_anomaly_model(df_clean, features_initial_dt)
    y_test_an, y_pred_an = svm_preds
    plot_feature_importance(feature_importance_svm.tail(10), 'Важность признаков для классификации аномалий простоя (Permutation Importance)', 'svm_feature_importance.png')

    # --- 12. Важность признаков для регрессионных моделей ---
    # Для Random Forest простоя
    rf_dt_model = models_dt['Random Forest'][0]
    feature_importance_dt = pd.Series(rf_dt_model.feature_importances_, index=X_train_dt.columns)
    plot_feature_importance(feature_importance_dt.sort_values(ascending=False).head(10), 'Важность признаков для прогнозирования простоя', 'feature_importance_downtime.png')

    # Для Random Forest парка
    rf_park_model = models_park['Random Forest'][0]
    feature_importance_park = pd.Series(rf_park_model.feature_importances_, index=X_train_park.columns)
    plot_feature_importance(feature_importance_park.sort_values(ascending=False).head(10), 'Важность признаков для прогнозирования рабочего парка', 'feature_importance_park.png')

    # --- 13. Временные паттерны аномалий ---
    plot_anomaly_temporal_analysis(df_clean)

    # --- 14. Сохранение результатов ---
    cluster_data_to_save = {'centers': centers_idx, 'clusters': clusters, 'ra': optimal_ra}
    save_models_and_results(
        models_dt, models_park, svm_dt, scaler_an_dt,
        lstm_model, scaler_lstm, cluster_data_to_save
    )

    # --- 15. Финальный вывод и отчет ---
    print("\n=== ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ АНАЛИЗА ===")
    
    # 1. КЛЮЧЕВЫЕ ЗАВИСИМОСТИ
    print("1. КЛЮЧЕВЫЕ ЗАВИСИМОСТИ:")
    correlation_with_downtime = corr_matrix['Простой факт'].abs().sort_values(ascending=False)
    correlation_with_park = corr_matrix['Рабочий парк'].abs().sort_values(ascending=False)
    
    print("  Наиболее сильные корреляции с простоем:")
    for feature, corr in correlation_with_downtime.head(6).items():
        if feature != 'Простой факт':
            print(f"   - {feature}: {corr:.3f}")
            
    print("\n  Наиболее сильные корреляции с рабочим парком:")
    for feature, corr in correlation_with_park.head(6).items():
        if feature != 'Рабочий парк':
            print(f"   - {feature}: {corr:.3f}")

    # 2. ЛУЧШИЕ МОДЕЛИ ПРОГНОЗИРОВАНИЯ
    print("\n2. ЛУЧШИЕ МОДЕЛИ ПРОГНОЗИРОВАНИЯ:")
    best_dt_model_name = max(metrics_dt, key=lambda k: metrics_dt[k][2]) # R2 is index 2
    best_dt_metrics = metrics_dt[best_dt_model_name]
    print(f"  Прогнозирование простоя:")
    print(f"   - Лучшая модель: {best_dt_model_name}")
    print(f"   - R²: {best_dt_metrics[2]:.4f}, MAE: {best_dt_metrics[0]:.2f}")

    best_park_model_name = max(metrics_park, key=lambda k: metrics_park[k][2])
    best_park_metrics = metrics_park[best_park_model_name]
    print(f"\n  Прогнозирование рабочего парка:")
    print(f"   - Лучшая модель: {best_park_model_name}")
    print(f"   - R²: {best_park_metrics[2]:.4f}, MAE: {best_park_metrics[0]:.2f}")

    print(f"\n  LSTM для прогнозирования простоя:")
    print(f"   - R²: {r2_lstm:.4f}, MAE: {mae_lstm:.2f}")

    # 3. АНАЛИЗ АНОМАЛИЙ
    print("\n3. АНАЛИЗ АНОМАЛИЙ:")
    anomaly_rate_dt = df_clean['Аномальный_простой'].mean()
    anomaly_rate_park = df_clean['Аномальный_парк'].mean()
    f1_svm = f1_score(y_test_an, y_pred_an)
    print(f"   - Аномалии простоя (> 15 часов): {anomaly_rate_dt:.1%} данных")
    print(f"   - Аномалии рабочего парка (> 5000 ваг.): {anomaly_rate_park:.1%} данных")
    print(f"   - SVM показал высокую точность в обнаружении аномалий простоя (F1-score: {f1_svm:.2f})")

    print("\n✅ АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
    print("📊 Результаты и модели сохранены в папке 'results/'")

if __name__ == '__main__':
    main()
