import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

train = pd.read_csv('../../bigfile/train_chungju_new.csv', encoding='utf-8')
test = pd.read_csv('../../bigfile/test_chungju.csv', encoding='utf-8')
train.shape
test.shape

# ---------------------------------------------------------------
# 전처리 베이스라인
train.columns = [col.split('(')[0].strip() for col in train.columns]
test.columns = [col.split('(')[0].strip() for col in test.columns]

train.isna().sum()/len(train)
test.isna().sum()/len(train)
train.info()
test.info()

# month/day를 month와 day로 분리하고 hour를 추가
train['일시'] = pd.to_datetime(train['일시'])
train['month'] = train['일시'].dt.month
train['day'] = train['일시'].dt.day
train['hour'] = train['일시'].dt.hour
train['day_of_year'] = train['일시'].dt.dayofyear

train['일조'] = train['일조'].fillna(0.0)
train['일사'] = train['일사'].fillna(0.0)
train['강수량'] = train['강수량'].fillna(0.0)
train['풍속'] = train['풍속'].fillna(train['풍속'].mean())

# month/day를 month와 day로 분리하고 hour를 추가
test['일시'] = pd.to_datetime(test['일시'])
test['month'] = test['일시'].dt.month
test['day'] = test['일시'].dt.day
test['hour'] = test['일시'].dt.hour
test['day_of_year'] = test['일시'].dt.dayofyear

test['일조'] = test['일조'].fillna(0.0)
test['일사'] = test['일사'].fillna(0.0)
test['강수량'] = test['강수량'].fillna(0.0)
# test 결측치 2개 있음 이슈 -> mean값으로 대체
test['습도'] = test['습도'].fillna(test['습도'].mean())

train.isna().sum()
test.isna().sum()

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
# LSTM 모델을 돌릴 때 일시(Date) 열을 인덱스로 설정하는 것이 일반적으로 좋습니다
train['일시'] = pd.to_datetime(train['일시'])
train.set_index('일시', inplace=True)

test['일시'] = pd.to_datetime(test['일시'])
test.set_index('일시', inplace=True)
# ----------------------------------------------------------------
# train, test data x/y 나누고 scaler 까지
# 2. X 변수와 Y 변수 설정
X_columns = ['기온', '강수량', '풍속', '습도', '일조', 'month', 'day', 'hour', '시간대별']
Y_column = ['일사']

import xgboost as XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# X와 Y 데이터 나누기
X_train = train[X_columns]
Y_train = train[Y_column].values.ravel()  # 1D 배열로 변환
X_test = test[X_columns]

# XGBoost 모델 설정
xgboost_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

# 모델 학습
xgboost_model.fit(X_train, Y_train)

# 예측
Y_pred = xgboost_model.predict(X_test)

# MAE 및 RMSE 계산
mae_value = mean_absolute_error(test[Y_column], Y_pred)
rmse_value = np.sqrt(mean_squared_error(test[Y_column], Y_pred))

# 결과 출력
print(f'MAE: {mae_value}')
print(f'RMSE: {rmse_value}')


































# XGBoost 모델 설정 및 학습
xg_model = XGBRegressor.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8
)

# 모델 학습
xg_model.fit(X_train_scaled, Y_train_scaled)

# 예측
test_predict = xg_model.predict(X_test_scaled)

# 스케일링 해제 (원래 값으로 복원)
test_predict_inv = scaler_Y.inverse_transform(test_predict.reshape(-1, 1))

# MAE 및 RMSE 계산
test_mae = mean_absolute_error(test[Y_column], test_predict_inv)
test_rmse = np.sqrt(mean_squared_error(test[Y_column], test_predict_inv))

# 결과 출력
print(f'Test MAE: {test_mae}')
print(f'Test RMSE: {test_rmse}')

# 예측값을 test DataFrame에 추가
test['예측_일사'] = test_predict_inv

# CSV 파일로 저장
output_path = '../../bigfile/test_with_predictions_xgboost.csv'
test.to_csv(output_path, encoding='utf-8-sig')