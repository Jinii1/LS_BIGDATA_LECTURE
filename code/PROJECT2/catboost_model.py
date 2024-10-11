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
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import math
from catboost import CatBoostRegressor

# 데이터 로드
train = pd.read_csv('../../bigfile/train_chungju_new.csv', encoding='utf-8')
test = pd.read_csv('../../bigfile/test_chungju.csv', encoding='utf-8')

real_test = pd.read_csv('../../bigfile/week2_data.csv', encoding='cp949')

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

train.info()

# '일시' 열을 인덱스로 변환
train.set_index('일시', inplace=True)
test.set_index('일시', inplace=True)

# X와 Y 데이터 설정
X_columns = ['기온', '강수량', '풍속', '습도', '일조', 'month', 'day', 'hour', '시간대별']
Y_column = ['일사']
# -----------------------------------------------------
# CatBoost 모델 설정 및 학습
cat_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.1,
    depth=6,
    loss_function='MAE',
    verbose=100,
    random_seed=42
)

# 모델 학습
cat_model.fit(train[X_columns], train[Y_column].values.ravel())  # Y 데이터는 1D로 변환

# 예측
test_predict = cat_model.predict(test[X_columns])

# MAE 및 RMSE 계산
test_mae = mean_absolute_error(test[Y_column], test_predict)
test_rmse = np.sqrt(mean_squared_error(test[Y_column], test_predict))

# 결과 출력
print(f'Test MAE: {test_mae}')
print(f'Test RMSE: {test_rmse}')

# Test MAE: 0.11084469389671316
# Test RMSE: 0.19564904346679596
# ----------------------------------------------------

