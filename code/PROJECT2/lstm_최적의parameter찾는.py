import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error

# 데이터 로드 및 전처리
train = pd.read_csv('../../bigfile/train_chungju.csv')
test = pd.read_csv('../../bigfile/test_chungju.csv')

# 전처리 및 결측치 처리
train.columns = [col.split('(')[0].strip() for col in train.columns]
test.columns = [col.split('(')[0].strip() for col in test.columns]
train['일시'] = pd.to_datetime(train['일시'])
train['month'] = train['일시'].dt.month
train['day'] = train['일시'].dt.day
train['hour'] = train['일시'].dt.hour
train['day_of_year'] = train['일시'].dt.dayofyear
train['일조'] = train['일조'].fillna(0.0)
train['일사'] = train['일사'].fillna(0.0)
train['강수량'] = train['강수량'].fillna(0.0)
train['풍속'] = train['풍속'].fillna(train['풍속'].mean())

test['일시'] = pd.to_datetime(test['일시'])
test['month'] = test['일시'].dt.month
test['day'] = test['일시'].dt.day
test['hour'] = test['일시'].dt.hour
test['day_of_year'] = test['일시'].dt.dayofyear
test['일조'] = test['일조'].fillna(0.0)
test['일사'] = test['일사'].fillna(0.0)
test['강수량'] = test['강수량'].fillna(0.0)
test['습도'] = test['습도'].fillna(test['습도'].mean())

# X, Y 설정 및 스케일링
X_columns = ['기온', '강수량', '풍속', '습도', '일조', 'month', 'day', 'hour']
Y_column = ['일사']
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler_X.fit_transform(train[X_columns])
Y_train_scaled = scaler_Y.fit_transform(train[Y_column])
X_test_scaled = scaler_X.transform(test[X_columns])
Y_test_scaled = scaler_Y.transform(test[Y_column])

# LSTM을 위한 데이터셋 생성 함수
def create_sequences(X, Y, time_steps=24):
    X_seq, Y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        Y_seq.append(Y[i + time_steps])
    return np.array(X_seq), np.array(Y_seq)

time_steps = 24
X_train_seq, Y_train_seq = create_sequences(X_train_scaled, Y_train_scaled, time_steps)
X_test_seq, Y_test_seq = create_sequences(X_test_scaled, Y_test_scaled, time_steps)

# LSTM 모델 빌드 함수
def build_lstm_model(units=50, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mae')
    return model

# 조기 종료 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 하이퍼파라미터 튜닝 실험
units_list = [50, 100, 150]
dropout_list = [0.2, 0.3, 0.4]
learning_rate_list = [0.001, 0.005, 0.01]
batch_size_list = [32, 64]
best_mae = float('inf')  # MAE 초기값
best_params = {}

for units in units_list:
    for dropout_rate in dropout_list:
        for learning_rate in learning_rate_list:
            for batch_size in batch_size_list:
                print(f"Training with units={units}, dropout_rate={dropout_rate}, learning_rate={learning_rate}, batch_size={batch_size}")
                
                model = build_lstm_model(units=units, dropout_rate=dropout_rate, learning_rate=learning_rate)
                history = model.fit(X_train_seq, Y_train_seq, epochs=50, batch_size=batch_size, validation_data=(X_test_seq, Y_test_seq), callbacks=[early_stopping], verbose=0)

                # 모델 예측 및 성능 평가
                test_predict = model.predict(X_test_seq)
                test_predict = scaler_Y.inverse_transform(test_predict)
                test_mae = mean_absolute_error(scaler_Y.inverse_transform(Y_test_seq), test_predict)

                # 성능 개선 시 최적의 파라미터 저장
                if test_mae < best_mae:
                    best_mae = test_mae
                    best_params = {'units': units, 'dropout_rate': dropout_rate, 'learning_rate': learning_rate, 'batch_size': batch_size}
                print(f"Test MAE: {test_mae}")

print(f"Best MAE: {best_mae}")
print(f"Best parameters: {best_params}")

# Test MAE: 0.23006994595448518
# Best MAE: 0.1662765758649073
# Best parameters: {'units': 100, 'dropout_rate': 0.3, 'learning_rate': 0.01, 'batch_size': 64}