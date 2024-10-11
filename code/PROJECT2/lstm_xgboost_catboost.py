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

# 3. 데이터 스케일링 (MinMaxScaler를 사용하여 0~1 범위로 정규화)
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))

# 학습 데이터 스케일링
X_train_scaled = scaler_X.fit_transform(train[X_columns])
Y_train_scaled = scaler_Y.fit_transform(train[Y_column])

# 테스트 데이터 스케일링
X_test_scaled = scaler_X.transform(test[X_columns])
Y_test_scaled = scaler_Y.transform(test[Y_column])

test.info()

# -> 같은 조건에서 모델을 돌리기 위해서 스케일링까지 전처리에 넣음
# ----------------------------------------------------------------
## LSTM 모델에 설명 추가한거라 . .
# 4. LSTM 모델을 위한 데이터셋 생성 함수
def create_sequences(X, Y, time_steps=24):
    X_seq, Y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])  # time_steps 길이만큼의 시퀀스 생성
        Y_seq.append(Y[i + time_steps])    # 해당 시퀀스의 다음 값을 예측
    return np.array(X_seq), np.array(Y_seq)

# 시퀀스 생성
time_steps = 24 # 몇 개의 과거 데이터를 참고할지를 결정하는 것
# 24 -> 하루의 최근 1일(24시간)의 데이터를 참고하여 미래를 예측
# 168 -> 최근 1주일(168시간)의 데이터를 참고하여 예측 -> 주간/주말 패턴 적합
X_train_seq, Y_train_seq = create_sequences(X_train_scaled, Y_train_scaled, time_steps)
X_test_seq, Y_test_seq = create_sequences(X_test_scaled, Y_test_scaled, time_steps)

# 5. LSTM 모델 구성
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
# unit: LSTM 레이어에 있는 뉴런의 수 -> 50차원의 내부 상태 유지하면서 데이터 학습
# 
model.add(Dropout(0.2))
# 20% 뉴런 임의로 비활성화 -> 과적합 방지
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# 6. 모델 컴파일
model.compile(optimizer='adam', loss='mae')

# 7. 모델 학습
history = model.fit(X_train_seq, Y_train_seq, epochs=10, batch_size=32, validation_data=(X_test_seq, Y_test_seq))

# 8. 모델 예측
train_predict = model.predict(X_train_seq)
test_predict = model.predict(X_test_seq)

# 9. 스케일링 해제 (원래 값으로 복원)
# lstm model 안정적으로 돌리기 위해서 scaling
# -> 모델 성능 정확하게 평가하기 위해선 실제값으로 돌려서 mae 계산해야 함
train_predict = scaler_Y.inverse_transform(train_predict)
test_predict = scaler_Y.inverse_transform(test_predict)
Y_train_seq_inv = scaler_Y.inverse_transform(Y_train_seq)
Y_test_seq_inv = scaler_Y.inverse_transform(Y_test_seq)

# 10. 성능 평가: MAE (Mean Absolute Error)
train_mae = mean_absolute_error(Y_train_seq_inv, train_predict)
test_mae = mean_absolute_error(Y_test_seq_inv, test_predict)

print(f'Test MAE: {test_mae}') 
# time_step 24 -> Test MAE: 0.1766000521511436

# mae: 예측 값과 실제 값 사이의 절대 오차 평균
# 이 값이 작을수록 예측 값이 실제 값에 가까워짐
# test data에서 예측한 '일사값'이 실제 '일사값'과 평균적으로 0.1898 차이남
# MAE를 다른 모델의 성능 지표나 베이스라인(기본 성능)과 비교할 때 유용
# 동일한 데이터에 대한 다른 모델의 mae와 비교해보면 모델이 얼마나 잘 예측했는지 알 수 있음

# 11. 시각화: 실제 값과 예측 값 비교 (테스트 데이터)
plt.figure(figsize=(14, 5))
plt.plot(Y_test_seq_inv, label='Actual')
plt.plot(test_predict, label='Predicted')
plt.title('Actual vs Predicted (Test Data)')
plt.xlabel('Time Steps')
plt.ylabel('일사')
plt.legend()
plt.show()

# --------------------------------------------
# XGBoost 사용
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import math

# 4. XGBoost 모델 학습
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)

# 모델 학습
xg_reg.fit(X_train_scaled, Y_train_scaled.ravel())  # .ravel()로 Y 데이터 1D로 변환

# 5. 모델 예측
test_predict = xg_reg.predict(X_test_scaled)

# 6. 스케일링 해제 (원래 값으로 복원)
test_predict_inv = scaler_Y.inverse_transform(test_predict.reshape(-1, 1))
Y_test_inv = scaler_Y.inverse_transform(Y_test_scaled)

# 7. 성능 평가: MAE, RMSE, SMAPE 계산
# MAE (Mean Absolute Error)
test_mae = mean_absolute_error(Y_test_inv, test_predict_inv)

# RMSE (Root Mean Squared Error)
test_rmse = math.sqrt(mean_squared_error(Y_test_inv, test_predict_inv))

# 결과 출력
print(f'Test MAE: {test_mae}')
print(f'Test RMSE: {test_rmse}')


# Test MAE: 0.11298962228941714
# Test RMSE: 0.19555020548184224

# ------------------------------------------------------------------
# Decision tree regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# 학습 데이터에서 train과 valid 데이터로 분할
X_train_split, X_valid, y_train_split, y_valid = train_test_split(X_train_scaled, Y_train_scaled, test_size=0.2, random_state=42)

# DecisionTreeRegressor 모델 생성 및 학습
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train_split, y_train_split)

# 검증 데이터에 대한 예측
valid_predict = model.predict(X_valid)

# 테스트 데이터에 대한 예측
test_predict = model.predict(X_test_scaled)

# 성능 평가: MAE 및 RMSE 계산

test_mae = mean_absolute_error(Y_test_scaled, test_predict)
test_rmse = math.sqrt(mean_squared_error(Y_test_scaled, test_predict))

# 결과 출력
print(f'Test MAE: {test_mae}, Test RMSE: {test_rmse}')
# Test MAE: 0.040428276573787406
# Test RMSE: 0.07597790937894469

# ------------------------------------------------------------------
## XGBoost ver.2
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# 2. XGBoost DMatrix 형태로 변환
dtrain = xgb.DMatrix(X_train_scaled, label=Y_train_scaled)
dtest = xgb.DMatrix(X_test_scaled, label=Y_test_scaled)

# 3. XGBoost 모델 하이퍼파라미터 설정
params = {
    'objective': 'reg:squarederror',  # 회귀 문제를 위한 목표 함수
    'max_depth': 5,                   # 트리의 최대 깊이 (적당히 큰 값으로 설정)
    'eta': 0.03,                      # 학습률 (더 작은 값으로 설정하여 학습 안정화)
    'subsample': 0.9,                 # 학습 데이터의 90%만 사용하여 학습
    'colsample_bytree': 0.8,          # 각 트리에서 80%의 피처만 사용
    'min_child_weight': 3,            # 최소 자식 노드 가중치 (값을 높여 과적합 방지)
    'lambda': 1.0,                    # L2 정규화 (기본값보다 높게 설정하여 과적합 방지)
    'alpha': 0.5,                     # L1 정규화 (과적합 방지에 도움)
    'eval_metric': 'mae',             # 평가 지표로 MAE 사용
    'seed': 42
}

# 4. 모델 학습 (early_stopping_rounds로 조기 종료)
model = xgb.train(params, dtrain, num_boost_round=1500, evals=[(dtest, 'test')], early_stopping_rounds=10, verbose_eval=False)

# 5. 모델 예측
train_predict = model.predict(dtrain)
test_predict = model.predict(dtest)

# 6. 스케일링 해제 (원래 값으로 복원)
train_predict_inv = scaler_Y.inverse_transform(train_predict.reshape(-1, 1))
test_predict_inv = scaler_Y.inverse_transform(test_predict.reshape(-1, 1))

# 7. 성능 평가: MAE 및 RMSE 계산
test_mae = mean_absolute_error(scaler_Y.inverse_transform(Y_test_scaled), test_predict_inv)

test_rmse = math.sqrt(mean_squared_error(scaler_Y.inverse_transform(Y_test_scaled), test_predict_inv))

# 결과 출력
print(f'Test MAE: {test_mae}, Test RMSE: {test_rmse}')
# Test MAE: 0.1174836649821338, Test RMSE: 0.19563548635607944
# -------------------------------------------------------------------------
## lstm parameter 적용 ver
# Create sequences
from tensorflow.keras.optimizers import Adam
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error

def create_sequences(X, Y, time_steps=24):
    X_seq, Y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        Y_seq.append(Y[i + time_steps])
    return np.array(X_seq), np.array(Y_seq)

time_steps = 24
X_train_seq, Y_train_seq = create_sequences(X_train_scaled, Y_train_scaled, time_steps)
X_test_seq, Y_test_seq = create_sequences(X_test_scaled, Y_test_scaled, time_steps)

# Build the LSTM model with best parameters
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(units=50))
model.add(Dense(units=1))

# Compile the model with learning rate = 0.01
optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mae')

# Train the model
history = model.fit(X_train_seq, Y_train_seq, epochs=10, batch_size=64, validation_data=(X_test_seq, Y_test_seq))

# Predict
train_predict = model.predict(X_train_seq)
test_predict = model.predict(X_test_seq)

# Inverse scaling
train_predict = scaler_Y.inverse_transform(train_predict)
test_predict = scaler_Y.inverse_transform(test_predict)
Y_train_seq_inv = scaler_Y.inverse_transform(Y_train_seq)
Y_test_seq_inv = scaler_Y.inverse_transform(Y_test_seq)

# Calculate MAE and RMSE
test_mae = mean_absolute_error(Y_test_seq_inv, test_predict)
test_rmse = math.sqrt(mean_squared_error(Y_test_seq_inv, test_predict))

print(f'Test MAE: {test_mae}, Test RMSE: {test_rmse}')
# Test MAE: 0.16323136748344833, Test RMSE: 0.29931949021057114

# 결과를 test DataFrame에 추가
test['예측_일사'] = test_predict_inv

# CSV 파일로 저장
output_path = '../../bigfile/test_with_predictions_catboost.csv'
test.to_csv(output_path, encoding='utf-8-sig')

# 예측한 y값 출력
predicted_values = test['예측_일사'].tolist()  # 예측값을 리스트로 변환
predicted_values[:10]  # 처음 10개 예측값 출력

# -------------------------------------------------------------------------
# CatBoost
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import math

# 4. CatBoost 모델 하이퍼파라미터 설정 및 학습
cat_model = CatBoostRegressor(
    iterations=2000,                # 더 많은 부스팅 라운드를 시도하여 성능 향상
    learning_rate=0.05,             # 낮은 학습률로 더 많은 학습 기회를 제공
    depth=8,                        # 트리 깊이를 늘려 좀 더 복잡한 패턴을 학습
    l2_leaf_reg=3,                  # L2 정규화 강도 (과적합 방지)
    loss_function='MAE',            # 손실 함수로 MAE 사용
    bagging_temperature=0.2,        # 샘플링 시 다소의 임의성을 부여해 과적합 방지
    random_strength=1.5,            # 스플릿 중 가중치를 랜덤하게 추가하여 다양성 증가
    verbose=100                     # 학습 과정 출력
)

# 모델 학습
cat_model.fit(X_train_scaled, Y_train_scaled)

# 5. 모델 예측
test_predict = cat_model.predict(X_test_scaled)

# 6. 스케일링 해제 (원래 값으로 복원)
test_predict_inv = scaler_Y.inverse_transform(test_predict.reshape(-1, 1))
Y_test_inv = scaler_Y.inverse_transform(Y_test_scaled)

# 7. 성능 평가: MAE, RMSE, SMAPE 계산
# MAE (Mean Absolute Error)
test_mae = mean_absolute_error(Y_test_inv, test_predict_inv)

# RMSE (Root Mean Squared Error)
test_rmse = math.sqrt(mean_squared_error(Y_test_inv, test_predict_inv))


# 결과 출력
print(f'Test MAE: {test_mae}')
print(f'Test RMSE: {test_rmse}')
# Test MAE: 0.10847676022824951
# Test RMSE: 0.19238614544627547
