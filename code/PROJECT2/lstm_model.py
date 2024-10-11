import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 데이터 로드
train = pd.read_csv('../../bigfile/train_chungju_new.csv', encoding='utf-8')
test = pd.read_csv('../../bigfile/test_chungju.csv', encoding='utf-8')

# 전처리 베이스라인
train.columns = [col.split('(')[0].strip() for col in train.columns]
test.columns = [col.split('(')[0].strip() for col in test.columns]

# 날짜 관련 정보 추가
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

# 시간대별 카테고리 추가
def get_time_of_day(hour):
    if 6 <= hour < 12:
        return 1  # Morning
    elif 12 <= hour < 18:
        return 2  # Afternoon
    elif 18 <= hour < 24:
        return 3  # Evening
    else:
        return 4  # Night

train['시간대별'] = train['hour'].apply(get_time_of_day)
test['시간대별'] = test['hour'].apply(get_time_of_day)

# '일시' 열을 인덱스로 변환
train.set_index('일시', inplace=True)
test.set_index('일시', inplace=True)

# X와 Y 데이터 설정
X_columns = ['기온', '강수량', '풍속', '습도', '일조', 'month', 'day', 'hour', '시간대별']
Y_column = ['일사']

# 데이터 스케일링 (MinMaxScaler를 사용하여 0~1 범위로 정규화)
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))

X_train_scaled = scaler_X.fit_transform(train[X_columns])
Y_train_scaled = scaler_Y.fit_transform(train[Y_column])

X_test_scaled = scaler_X.transform(test[X_columns])
Y_test_scaled = scaler_Y.transform(test[Y_column])

# LSTM 모델 구축
def create_sequences(X, Y, time_steps=24):
    X_seq, Y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        Y_seq.append(Y[i + time_steps])
    return np.array(X_seq), np.array(Y_seq)

time_steps = 24
X_train_seq, Y_train_seq = create_sequences(X_train_scaled, Y_train_scaled, time_steps)
X_test_seq, Y_test_seq = create_sequences(X_test_scaled, Y_test_scaled, time_steps)

model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(units=50))
model.add(Dense(units=1))

# 모델 컴파일
optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mae')

# 모델 학습
history = model.fit(X_train_seq, Y_train_seq, epochs=10, batch_size=64, validation_data=(X_test_seq, Y_test_seq))

# 예측
train_predict = model.predict(X_train_seq)
test_predict = model.predict(X_test_seq)

# 스케일링 해제
train_predict = scaler_Y.inverse_transform(train_predict)
test_predict = scaler_Y.inverse_transform(test_predict)
Y_train_seq_inv = scaler_Y.inverse_transform(Y_train_seq)
Y_test_seq_inv = scaler_Y.inverse_transform(Y_test_seq)

# MAE 및 RMSE 계산
test_mae = mean_absolute_error(Y_test_seq_inv, test_predict)
test_rmse = math.sqrt(mean_squared_error(Y_test_seq_inv, test_predict))

# 결과 출력
print(f'Test MAE: {test_mae}')
print(f'Test RMSE: {test_rmse}')

# 결과를 test DataFrame에 추가
test['예측_일사'] = test_predict  # test_predict를 추가

test['일사'].shape
test_predict.shape

# CSV 파일로 저장
output_path = '../../bigfile/test_with_predictions_lstm.csv'
test.to_csv(output_path, encoding='utf-8-sig')