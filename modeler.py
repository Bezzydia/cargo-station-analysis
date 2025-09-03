"""Модуль для построения и оценки моделей."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, classification_report
from sklearn.inspection import permutation_importance
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os

# Установка seed для воспроизводимости (лучше делать в main)
# np.random.seed(42)
# tf.random.set_seed(42) 

def prepare_modeling_data(df_clean: pd.DataFrame):
    """
    Готовит данные для моделирования, разделяя признаки и целевые переменные.
    """
    print("Подготовка данных для моделирования...")
    # Для модели простоя исключаем компоненты простоя и рабочий парк (если он используется для расчета)
    features_initial_dt = [
        't(1) - Прибытие', 't(p) - Расформирование', 't(c) - Сортировка', 't(o) - Отправление',
        'Прибытие общее', 'Отправление общее',
        'Прибытие с запада', 'Отправление на восток',
        'Прибытие с востока', 'Отправление на запад',
        'Рабочий парк нечетный', 'Рабочий парк четный',
        'дисбаланс_направлений'
    ]
    X_dt = df_clean[features_initial_dt]
    y_dt = df_clean['Простой факт']

    # Для модели рабочего парка исключаем компоненты рабочего парка
    features_initial_park = [
        't(1) - Прибытие', 't(p) - Расформирование', 't(c) - Сортировка', 't(o) - Отправление',
        'Прибытие общее', 'Отправление общее',
        'Прибытие с запада', 'Отправление на восток',
        'Прибытие с востока', 'Отправление на запад',
        'Простой без переработки факт'
    ]
    X_park = df_clean[features_initial_park]
    y_park = df_clean['Рабочий парк']

    # Проверка на пропущенные значения и удаление
    print("Пропущенные значения в признаках для простоя:")
    print(X_dt.isnull().sum())
    print("\nПропущенные значения в признаках для парка:")
    print(X_park.isnull().sum())

    X_dt_clean = X_dt.dropna()
    y_dt_clean = y_dt.loc[X_dt_clean.index]
    X_park_clean = X_park.dropna()
    y_park_clean = y_park.loc[X_park_clean.index]

    # Разделение данных с сохранением временной последовательности
    split_idx_dt = int(len(X_dt_clean) * 0.8)
    X_train_dt, X_test_dt = X_dt_clean.iloc[:split_idx_dt], X_dt_clean.iloc[split_idx_dt:]
    y_train_dt, y_test_dt = y_dt_clean.iloc[:split_idx_dt], y_dt_clean.iloc[split_idx_dt:]

    split_idx_park = int(len(X_park_clean) * 0.8)
    X_train_park, X_test_park = X_park_clean.iloc[:split_idx_park], X_park_clean.iloc[split_idx_park:]
    y_train_park, y_test_park = y_park_clean.iloc[:split_idx_park], y_park_clean.iloc[split_idx_park:]

    print(f"\nРазмер обучающей выборки для простоя: {X_train_dt.shape}")
    print(f"Размер тестовой выборки для простоя: {X_test_dt.shape}")
    print(f"Размер обучающей выборки для парка: {X_train_park.shape}")
    print(f"Размер тестовой выборки для парка: {X_test_park.shape}")
    
    return (X_train_dt, X_test_dt, y_train_dt, y_test_dt, 
            X_train_park, X_test_park, y_train_park, y_test_park)

def evaluate_model(y_true, y_pred, model_name):
    """Оценивает модель по метрикам."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name}:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    return mae, rmse, r2

def train_and_evaluate_regression_models(X_train_dt, X_test_dt, y_train_dt, y_test_dt,
                                         X_train_park, X_test_park, y_train_park, y_test_park):
    """
    Обучает и оценивает модели регрессии для простоя и рабочего парка.
    """
    print("\n=== МОДЕЛИ ДЛЯ ПРОСТОЯ ===")
    # Нормализация данных для простоя
    scaler_dt = StandardScaler()
    X_train_dt_scaled = scaler_dt.fit_transform(X_train_dt)
    X_test_dt_scaled = scaler_dt.transform(X_test_dt)

    models_dt = {}
    predictions_dt = {}
    metrics_dt = {}

    # Ridge регрессия
    ridge_dt = Ridge(alpha=1.0)
    ridge_dt.fit(X_train_dt_scaled, y_train_dt)
    y_pred_dt_ridge = ridge_dt.predict(X_test_dt_scaled)
    metrics_dt['Ridge'] = evaluate_model(y_test_dt, y_pred_dt_ridge, "Ridge регрессия")
    models_dt['Ridge'] = (ridge_dt, scaler_dt)

    # Random Forest
    rf_dt = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_dt.fit(X_train_dt, y_train_dt)
    y_pred_dt_rf = rf_dt.predict(X_test_dt)
    metrics_dt['Random Forest'] = evaluate_model(y_test_dt, y_pred_dt_rf, "Random Forest")
    models_dt['Random Forest'] = (rf_dt, None) # RF не требует внешнего скалера

    # Gradient Boosting
    gb_dt = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_dt.fit(X_train_dt, y_train_dt)
    y_pred_dt_gb = gb_dt.predict(X_test_dt)
    metrics_dt['Gradient Boosting'] = evaluate_model(y_test_dt, y_pred_dt_gb, "Gradient Boosting")
    models_dt['Gradient Boosting'] = (gb_dt, None)

    print("\n=== МОДЕЛИ ДЛЯ РАБОЧЕГО ПАРКА ===")
    # Нормализация данных для рабочего парка
    scaler_park = StandardScaler()
    X_train_park_scaled = scaler_park.fit_transform(X_train_park)
    X_test_park_scaled = scaler_park.transform(X_test_park)

    models_park = {}
    predictions_park = {}
    metrics_park = {}

    # Ridge регрессия
    ridge_park = Ridge(alpha=1.0)
    ridge_park.fit(X_train_park_scaled, y_train_park)
    y_pred_park_ridge = ridge_park.predict(X_test_park_scaled)
    metrics_park['Ridge'] = evaluate_model(y_test_park, y_pred_park_ridge, "Ridge регрессия")
    models_park['Ridge'] = (ridge_park, scaler_park)

    # Random Forest
    rf_park = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_park.fit(X_train_park, y_train_park)
    y_pred_park_rf = rf_park.predict(X_test_park)
    metrics_park['Random Forest'] = evaluate_model(y_test_park, y_pred_park_rf, "Random Forest")
    models_park['Random Forest'] = (rf_park, None)

    # Gradient Boosting
    gb_park = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_park.fit(X_train_park, y_train_park)
    y_pred_park_gb = gb_park.predict(X_test_park)
    metrics_park['Gradient Boosting'] = evaluate_model(y_test_park, y_pred_park_gb, "Gradient Boosting")
    models_park['Gradient Boosting'] = (gb_park, None)

    return models_dt, models_park, metrics_dt, metrics_park

def time_series_cv(model, X, y, model_name, cv_splits=5):
    """Выполняет временную кросс-валидацию."""
    print(f"Кросс-валидация для {model_name}...")
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
    mean_score = scores.mean()
    std_score = scores.std()
    print(f"  {model_name}: {mean_score:.4f} (+/- {std_score * 2:.4f})")
    return mean_score, std_score

def train_lstm_model(df_clean, lstm_features, sequence_length=24, epochs=10):
    """
    Обучает модель LSTM для прогнозирования простоя.
    """
    print("\n=== LSTM ДЛЯ ПРОГНОЗИРОВАНИЯ ПРОСТОЯ ===")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    
    # Нормализация данных
    scaler_lstm = MinMaxScaler()
    lstm_data = df_clean[lstm_features].values
    scaled_lstm_data = scaler_lstm.fit_transform(lstm_data)

    # Создание последовательностей
    def create_sequences(data, seq_length, target_col_idx):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length, target_col_idx])
        return np.array(X), np.array(y)

    X_lstm, y_lstm = create_sequences(scaled_lstm_data, sequence_length, lstm_features.index('Простой факт'))

    # Разделение на обучающую и тестовую выборки
    train_size = int(0.8 * len(X_lstm))
    X_train_lstm, X_test_lstm = X_lstm[:train_size], X_lstm[train_size:]
    y_train_lstm, y_test_lstm = y_lstm[:train_size], y_lstm[train_size:]

    print(f"Размер обучающей выборки для LSTM: {X_train_lstm.shape}")
    print(f"Размер тестовой выборки для LSTM: {X_test_lstm.shape}")

    # Построение модели LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    history = model.fit(
        X_train_lstm, y_train_lstm,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    # Оценка модели
    train_predict = model.predict(X_train_lstm)
    test_predict = model.predict(X_test_lstm)

    # Обратная нормализация
    temp_train = np.zeros((len(train_predict), scaled_lstm_data.shape[1]))
    temp_train[:, lstm_features.index('Простой факт')] = train_predict[:, 0]
    temp_test = np.zeros((len(test_predict), scaled_lstm_data.shape[1]))
    temp_test[:, lstm_features.index('Простой факт')] = test_predict[:, 0]

    train_predict_inv = scaler_lstm.inverse_transform(temp_train)[:, lstm_features.index('Простой факт')]
    test_predict_inv = scaler_lstm.inverse_transform(temp_test)[:, lstm_features.index('Простой факт')]

    y_train_inv = scaled_lstm_data[sequence_length:train_size+sequence_length, lstm_features.index('Простой факт')]
    y_test_inv = scaled_lstm_data[train_size+sequence_length:train_size+sequence_length+len(y_test_lstm), lstm_features.index('Простой факт')]

    # Метрики
    mae_lstm = mean_absolute_error(y_test_inv, test_predict_inv)
    rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, test_predict_inv))
    r2_lstm = r2_score(y_test_inv, test_predict_inv)
    print(f"\nLSTM для прогнозирования простоя:")
    print(f"MAE: {mae_lstm:.2f} часов")
    print(f"RMSE: {rmse_lstm:.2f} часов")
    print(f"R²: {r2_lstm:.4f}")
    
    return model, history, (y_test_inv, test_predict_inv), (mae_lstm, rmse_lstm, r2_lstm), scaler_lstm

def train_svm_anomaly_model(df_clean, features_initial_dt):
    """
    Обучает SVM модель для классификации аномалий простоя.
    """
    print("\n=== SVM ДЛЯ КЛАССИФИКАЦИИ АНОМАЛИЙ ПРОСТОЯ ===")
    # Подготовка данных для классификации
    X_anomaly_dt = df_clean[features_initial_dt]
    y_anomaly_dt = df_clean['Аномальный_простой']

    # Удаляем строки с пропущенными значениями
    X_anomaly_dt = X_anomaly_dt.dropna()
    y_anomaly_dt = y_anomaly_dt.loc[X_anomaly_dt.index]

    # Разделение данных
    # Используем shuffle=False для временных рядов
    split_idx = int(0.8 * len(X_anomaly_dt))
    X_train_dt_an, X_test_dt_an = X_anomaly_dt.iloc[:split_idx], X_anomaly_dt.iloc[split_idx:]
    y_train_dt_an, y_test_dt_an = y_anomaly_dt.iloc[:split_idx], y_anomaly_dt.iloc[split_idx:]

    # Нормализация
    scaler_an_dt = StandardScaler()
    X_train_dt_an_scaled = scaler_an_dt.fit_transform(X_train_dt_an)
    X_test_dt_an_scaled = scaler_an_dt.transform(X_test_dt_an)

    # Обучение SVM
    svm_dt = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    svm_dt.fit(X_train_dt_an_scaled, y_train_dt_an)

    # Предсказание и оценка
    y_pred_dt_an = svm_dt.predict(X_test_dt_an_scaled)
    # y_prob_dt_an = svm_dt.predict_proba(X_test_dt_an_scaled)[:, 1] # Не используется, но можно сохранить

    print("\nМетрики классификации аномалий простоя:")
    print(classification_report(y_test_dt_an, y_pred_dt_an))
    print(f"F1-score: {f1_score(y_test_dt_an, y_pred_dt_an):.2f}")
    
    # Вычисляем важность признаков через permutation
    result = permutation_importance(
        svm_dt,
        X_test_dt_an_scaled,
        y_test_dt_an,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    # Сортируем признаки по важности
    sorted_idx = result.importances_mean.argsort()
    feature_importance_svm = pd.Series(
        result.importances_mean[sorted_idx],
        index=[X_train_dt_an.columns[i] for i in sorted_idx]
    )
    
    return svm_dt, scaler_an_dt, (y_test_dt_an, y_pred_dt_an), feature_importance_svm

def save_models_and_results(models_dt, models_park, svm_dt, scaler_an_dt,
                            lstm_model, scaler_lstm, cluster_data, results_dir="results"):
    """
    Сохраняет обученные модели и результаты.
    """
    print("\nСохранение моделей и результатов...")
    models_path = os.path.join(results_dir, "models")
    os.makedirs(models_path, exist_ok=True)
    
    # Сохранение моделей регрессии для простоя
    for name, (model, scaler) in models_dt.items():
        joblib.dump(model, os.path.join(models_path, f'{name.lower().replace(" ", "_")}_downtime_model.pkl'))
        if scaler:
             joblib.dump(scaler, os.path.join(models_path, f'scaler_downtime_{name.lower().replace(" ", "_")}.pkl'))

    # Сохранение моделей регрессии для парка
    for name, (model, scaler) in models_park.items():
        joblib.dump(model, os.path.join(models_path, f'{name.lower().replace(" ", "_")}_park_model.pkl'))
        if scaler:
             joblib.dump(scaler, os.path.join(models_path, f'scaler_park_{name.lower().replace(" ", "_")}.pkl'))

    # Сохранение SVM для аномалий
    joblib.dump(svm_dt, os.path.join(models_path, 'svm_anomaly_model.pkl'))
    joblib.dump(scaler_an_dt, os.path.join(models_path, 'scaler_anomaly.pkl'))

    # Сохранение LSTM (если используется Keras, можно сохранить веса и архитектуру)
    # lstm_model.save(os.path.join(models_path, 'lstm_model.h5')) # Требует h5py
    # Для простоты сохраним сам объект модели (может быть не лучшей практикой для больших моделей)
    joblib.dump(lstm_model, os.path.join(models_path, 'lstm_model.pkl'))
    joblib.dump(scaler_lstm, os.path.join(models_path, 'scaler_lstm.pkl'))

    # Сохранение кластеризации
    joblib.dump(cluster_data, os.path.join(models_path, 'subtractive_clustering.pkl'))
    
    print("Модели и результаты сохранены.")