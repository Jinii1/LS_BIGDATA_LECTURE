# ! pip install tensorflow numpy pandas matplotlib scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import seaborn as sns
import datetime


# 데이터 로드
train = pd.read_csv('../../bigfile/train_chungju.csv')
test = pd.read_csv('../../bigfile/test_chungju.csv')


# ------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# 옵션 1: CSV 파일에서 주가 데이터 로드
data = pd.read_csv('AAPL.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data[['Close']]

# 옵션 2: 외부 소스에서 주가 데이터 가져오기
# ticker = 'AAPL'
# data = yf.download(ticker, start='2020-01-01', end='2023-01-01')
# data = data[['Close']]

# 데이터 플롯
plt.figure(figsize=(14, 5))
plt.plot(data)
plt.title('시간에 따른 주가')
plt.xlabel('날짜')
plt.ylabel('가격')
plt.show()

# 데이터 정규화
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# LSTM을 위한 데이터셋 생성
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# 입력 데이터를 [샘플, 타임 스텝, 특징] 형태로 재구성
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# LSTM 모델 구축
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 요약 출력
model.summary()

# 학습 손실 플롯
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.title('모델 학습 손실')
plt.xlabel('에포크')
plt.ylabel('손실')
plt.show()

# 학습 데이터와 테스트 데이터를 사용하여 예측
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 예측 값의 변환 복원
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])

# 결과 플롯
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))
plt.plot(data.index, scaler.inverse_transform(scaled_data), label='Original Data')
plt.plot(data.index[time_step:len(train_predict) + time_step], train_predict, label='Train Predict')
plt.plot(data.index[len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1], test_predict, label='Test Predict')
plt.title('주가 예측')
plt.xlabel('날짜')
plt.ylabel('주가')
plt.legend()
plt.show()

