import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 데이터 로드
train = pd.read_csv('../../bigfile/train_chungju_new.csv', encoding='utf-8')
test = pd.read_csv('../../bigfile/test_chungju.csv', encoding='utf-8')
week2 = pd.read_csv('../../bigfile/week2_data.csv',encoding='cp949')



# 결측치 확인
train.isna().sum()
test.isna().sum()

# train 전처리
train.columns = [col.split('(')[0].strip() for col in train.columns]

train['일시'] = pd.to_datetime(train['일시'])
train['year'] = train['일시'].dt.year
train['month'] = train['일시'].dt.month
train['hour'] = train['일시'].dt.hour

# 결측치 처리
train['일조'] = train['일조'].fillna(0.0)
train['일사'] = train['일사'].fillna(0.0)
train['강수량'] = train['강수량'].fillna(0.0)
train['풍속'] = train['풍속'].fillna(train['풍속'].mean())

# test 전처리
test.columns = [col.split('(')[0].strip() for col in test.columns]

test['일시'] = pd.to_datetime(test['일시'])
test['year'] = test['일시'].dt.year
test['month'] = test['일시'].dt.month
test['hour'] = test['일시'].dt.hour

test['일조'] = test['일조'].fillna(0.0)
test['일사'] = test['일사'].fillna(0.0)
test['강수량'] = test['강수량'].fillna(0.0)
test['습도'] = test['습도'].fillna(test['습도'].mean())

train.isna().sum()
test.isna().sum()

##############################
# Feature와 Target 분리
X_train = train.drop(['일시', '일사'], axis=1)
y_train = train['일사']
X_test = test.drop(['일시', '일사'], axis=1)
y_test = test['일사']

##############################
# 모델 학습 및 예측 함수
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, is_lstm=False):
    if is_lstm:
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        test_predict = model.predict(X_test).flatten()
    else:
        model.fit(X_train, y_train)
        test_predict = model.predict(X_test)
    
    train_predict = model.predict(X_train) if not is_lstm else model.predict(X_train).flatten()
    
    mae_train = mean_absolute_error(y_train, train_predict)
    rmse_train = np.sqrt(mean_squared_error(y_train, train_predict))
    mae_test = mean_absolute_error(y_test, test_predict)
    rmse_test = np.sqrt(mean_squared_error(y_test, test_predict))
    
    print(f"Train - MAE: {mae_train:.5f}, RMSE: {rmse_train:.5f}")
    print(f"Test - MAE: {mae_test:.5f}, RMSE: {rmse_test:.5f}")
    return test_predict

# 데이터 준비 (train, test를 위 코드와 동일하게 전처리했다고 가정)
# Linear Regression
print("Linear Regression Results")
linear_model = LinearRegression()
linear_pred = train_and_evaluate_model(linear_model, X_train, y_train, X_test, y_test)

# Decision Tree Regressor
print("\nDecision Tree Regressor Results")
tree_model = DecisionTreeRegressor(random_state=42)
tree_pred = train_and_evaluate_model(tree_model, X_train, y_train, X_test, y_test)

# Random Forest Regressor
print("\nRandom Forest Regressor Results")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_pred = train_and_evaluate_model(rf_model, X_train, y_train, X_test, y_test)

# Gradient Boosting Regressor
print("\nGradient Boosting Regressor Results")
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_pred = train_and_evaluate_model(gb_model, X_train, y_train, X_test, y_test)

# LightGBM Regressor
print("\nLightGBM Regressor Results")
lgbm_model = LGBMRegressor(n_estimators=100, random_state=42)
lgbm_pred = train_and_evaluate_model(lgbm_model, X_train, y_train, X_test, y_test)

# CatBoost Regressor
print("\nCatBoost Regressor Results")
catboost_model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, random_seed=42,verbose=100)
catboost_pred = train_and_evaluate_model(catboost_model, X_train, y_train, X_test, y_test)

# LSTM Model
def create_sequences(X, Y, time_step=1):
    X_seq, Y_seq = [], []
    for i in range(len(X) - time_step):
        X_seq.append(X[i:(i + time_step)])
        Y_seq.append(Y[i + time_step])
    return np.array(X_seq), np.array(Y_seq)

X_seq_train, y_seq_train = create_sequences(X_train, y_train, time_step=24)
X_seq_test, y_seq_test = create_sequences(X_test, y_test, time_step=24)

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# LSTM 모델 학습을 위한 데이터 준비
# 이미 (samples, time_steps, features) 형태이므로 reshape 불필요
lstm_model = create_lstm_model((X_seq_train.shape[1], X_seq_train.shape[2]))
print("\nLSTM Results")
lstm_pred = train_and_evaluate_model(lstm_model, X_seq_train, y_seq_train, X_seq_test, y_seq_test, is_lstm=True)


# 예측값과 실제값 비교 시각화
plt.figure(figsize=(14, 7))
plt.plot(test['일시'], y_test, label='Actual', color='blue', alpha=0.6)
plt.plot(test['일시'], lgbm_pred, label='LightGBM Predicted', color='orange', alpha=0.6)
plt.title('Actual vs LightGBM Predicted')
plt.xlabel('Date')
plt.ylabel('Solar Radiation (일사)')
plt.legend()
plt.show()

#############

# Linear Regression Results
# Test - MAE: 0.41062, RMSE: 0.53094

# Decision Tree Regressor Results
# Test - MAE: 0.12740, RMSE: 0.23277

# Random Forest Regressor Results
# Test - MAE: 0.09042, RMSE: 0.16362

# Gradient Boosting Regressor Results
# Test - MAE: 0.14618, RMSE: 0.21486

# LightGBM Regressor Results
# Test - MAE: 0.09005, RMSE: 0.15402

# CatBoost Regressor
# Test - MAE: 0.09896, RMSE: 0.15659

# LSTM Results
# Test - MAE: 17.42467, RMSE: 17.84994
# 너무 높아서 다시 돌려볼 예정

###########################################

df = pd.read_csv('../../bigfile/data_week2.csv',encoding='euc-kr')
df.columns
df.columns = [col.split('(')[0].strip() for col in df.columns]
df.rename(columns={'date_time': '일시'}, inplace=True)

df['일시'] = pd.to_datetime(df['일시'])
df['year'] = df['일시'].dt.year
df['month'] = df['일시'].dt.month
df['hour'] = df['일시'].dt.hour

df.isna().sum()


# LightGBM Regressor
print("\nLightGBM Regressor Results")
lgbm_model = LGBMRegressor(n_estimators=100, random_state=42)
lgbm_pred = train_and_evaluate_model(lgbm_model, X_train, y_train, X_test, y_test)

X_df = df[['기온', '강수량', '풍속', '습도', '일조', 'year', 'month', 'hour']]
y_pred = np.round(lgbm_model.predict(X_df), 2)
y_pred

# 일사(MJ/m2)
df['예측일사()'] = y_pred

df.columns
df_new = df[['num','일시','전력사용량','비전기냉방설비운영',
       '태양광보유','예측일사']]

df_new.groubpy('')


# 08시-18시 0 초과인거


len(train.query('hour>7 & hour<19 & 일사>0')) # 13417
len(train.query('hour>7 & hour<19 & 일사==0')) # 300
len(train.query('(hour<8 or hour>18) & 일사>0')) # 2057
len(train.query('(hour<8 or hour>18) & 일사==0')) # 14153

# 낮과 밤을 구분하여 새로운 열 추가
train['time_of_day'] = train['hour'].apply(lambda x: 'Day' if 8 <= x <= 18 else 'Night')

# 일사가 0인 경우와 0 초과인 경우를 구분하는 열 추가
train['일사_status'] = train['일사'].apply(lambda x: '일사 > 0' if x > 0 else '일사 = 0')