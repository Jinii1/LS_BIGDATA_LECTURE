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

train = pd.read_csv('../../bigfile/train_chungju.csv')
test = pd.read_csv('../../bigfile/test_chungju.csv')
# ---------------------------------------------------------------
# 전처리 베이스라인
train.columns = [col.split('(')[0].strip() for col in train.columns]
test.columns = [col.split('(')[0].strip() for col in test.columns]

train.isna().sum()/len(train)
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

# '일시' 열을 인덱스로 변환
# LSTM 모델을 돌릴 때 일시(Date) 열을 인덱스로 설정하는 것이 일반적으로 좋습니다
train['일시'] = pd.to_datetime(train['일시'])
train.set_index('일시', inplace=True)

test['일시'] = pd.to_datetime(test['일시'])
test.set_index('일시', inplace=True)
# ----------------------------------------------------------------
# train, test data x/y 나누고 scaler 까지
# 2. X 변수와 Y 변수 설정
X_columns = ['기온', '강수량', '풍속', '습도', '일조', 'month', 'day', 'hour']
Y_column = ['일사']

# 3. 데이터 스케일링 (MinMaxScaler를 사용하여 0~1 범위로 정규화)
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))

# 학습 데이터 스케일링
X_train_scaled = scaler_X.fit_transform(train[X_columns])
Y_train_scaled = scaler_Y.fit_transform(train[Y_column])

# 테스트 데이터 스케일링
X_test_scaled = scaler_X.transform(test[X_columns])
Y_test_scaled = scaler_Y.transform(test[Y_column])

# -> 같은 조건에서 모델을 돌리기 위해서 스케일링까지 전처리에 넣음
# ----------------------------------------------------------------
# linear regression
from sklearn.linear_model import LinearRegression

# 2. Linear Regression 모델 학습
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train_scaled, Y_train_scaled)

# 3. 모델 예측
train_predict_linear = linear_reg_model.predict(X_train_scaled)
test_predict_linear = linear_reg_model.predict(X_test_scaled)

# 4. 성능 평가: MAE (Mean Absolute Error)
train_mae_linear = mean_absolute_error(Y_train_scaled, train_predict_linear)
test_mae_linear = mean_absolute_error(Y_test_scaled, test_predict_linear)

print(f'Linear Regression Test MAE: {test_mae_linear}') # Linear Regression Test MAE: 0.10353570520688135
# ----------------------------------------------------------------
real_test = pd.read_csv('../../bigfile/week2_data.csv', encoding='cp949')
real_test

# 전처리 베이스라인
real_test.columns = [col.split('(')[0].strip() for col in real_test.columns]
real_test

real_test['date_time'] = pd.to_datetime(real_test['date_time'])
real_test['month'] = real_test['date_time'].dt.month
real_test['day'] = real_test['date_time'].dt.day
real_test['hour'] = real_test['date_time'].dt.hour
real_test['day_of_year'] = real_test['date_time'].dt.dayofyear

# '일시' 열을 인덱스로 변환
# LSTM 모델을 돌릴 때 일시(Date) 열을 인덱스로 설정하는 것이 일반적으로 좋습니다
real_test['date_time'] = pd.to_datetime(real_test['date_time'])
real_test.set_index('date_time', inplace=True)

real_test.columns

# train, test data x/y 나누고 scaler 까지
# 2. X 변수와 Y 변수 설정
real_X_columns = ['기온', '강수량', '풍속', '습도', '일조', 'month', 'day', 'hour']

# 3. 데이터 스케일링 (MinMaxScaler를 사용하여 0~1 범위로 정규화)
scaler_X = MinMaxScaler(feature_range=(0, 1))

# 학습 데이터 스케일링
real_X_train_scaled = scaler_X.fit_transform(train[X_columns])

# 테스트 데이터 스케일링
real_X_test_scaled = scaler_X.transform(test[X_columns])
# -> 같은 조건에서 모델을 돌리기 위해서 스케일링까지 전처리에 넣음
# ----------------------------------------------------------------
# real_test 데이터를 기반으로 예측 수행
real_test_predict = linear_reg_model.predict(real_X_test_scaled)

real_test.columns
real_X_test_scaled.shape

# 스케일링 해제 (원래 값으로 복원)
real_test_predict_inverse = scaler_Y.inverse_transform(real_test_predict)
real_predict=real_test_predict_inverse
real_predict.shape
# --------------------------
# X, Y 컬럼 설정
X_columns = ['기온', '강수량', '풍속', '습도', '일조', 'month', 'day', 'hour']
Y_column = ['일사']

# 데이터 스케일링
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))

# 학습 데이터 스케일링
X_train_scaled = scaler_X.fit_transform(train[X_columns])
Y_train_scaled = scaler_Y.fit_transform(train[Y_column])

# real_test 데이터에서 필요한 8개의 특성 추출 (기온, 풍속, 습도, 강수량, 일조, month, day, hour)
real_X = real_test[['기온', '풍속', '습도', '강수량', '일조', 'month', 'day', 'hour']]

# 2040개의 row씩 60개의 set으로 분할
sets = np.array_split(real_X, 60)

# Linear Regression 모델 학습
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train_scaled, Y_train_scaled)

# 각 set에 대해 모델 예측 수행 및 결과 저장
predictions = []
for data_set in sets:
    # 스케일링 적용
    data_set_scaled = scaler_X.transform(data_set)
    
    # 예측 수행
    pred_scaled = linear_reg_model.predict(data_set_scaled)
    
    # 스케일링 해제 (원래 값으로 복원)
    pred = scaler_Y.inverse_transform(pred_scaled)
    
    # 예측 결과 저장
    predictions.append(pred)
