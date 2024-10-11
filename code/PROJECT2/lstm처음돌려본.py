import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# 데이터 로드
train = pd.read_csv('../../bigfile/train_chungju.csv')
test = pd.read_csv('../../bigfile/test_chungju.csv')

# 전처리: 컬럼 이름 수정 및 결측값 처리
train.columns = [col.split('(')[0].strip() for col in train.columns]
train['일시'] = pd.to_datetime(train['일시'])
train['month'] = train['일시'].dt.month
train['day'] = train['일시'].dt.day
train['hour'] = train['일시'].dt.hour
train['day_of_year'] = train['일시'].dt.dayofyear

# 결측값 0으로 대체
train['일조'] = train['일조'].fillna(0.0)
train['일사'] = train['일사'].fillna(0.0)
train['강수량'] = train['강수량'].fillna(0.0)
train['풍속'] = train['풍속'].fillna(0.0)

# X 변수와 Y 변수 설정
X_columns = ['기온', '강수량', '풍속', '습도', '일조', 'month', 'day', 'hour']
Y_column = ['일사']

# 데이터 스케일링
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))

# X, Y 스케일링
X_scaled = scaler_X.fit_transform(train[X_columns])
Y_scaled = scaler_Y.fit_transform(train[Y_column])

# LSTM 모델을 위한 데이터셋 생성 함수
def create_sequences(X, Y, time_steps=60):
    X_seq, Y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])  # time_steps 길이만큼의 시퀀스 생성
        Y_seq.append(Y[i + time_steps])    # 해당 시퀀스의 다음 값을 예측
    return np.array(X_seq), np.array(Y_seq)

# 시퀀스 생성
time_steps = 60
X_seq, Y_seq = create_sequences(X_scaled, Y_scaled, time_steps)

# 훈련 및 테스트 데이터 분리 (80%를 훈련 데이터로 사용)
train_size = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
Y_train, Y_test = Y_seq[:train_size], Y_seq[train_size:]

# LSTM 모델 구성
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_test, Y_test))

# 예측 및 성능 평가
predictions = model.predict(X_test)
predictions = scaler_Y.inverse_transform(predictions)  # 정규화 해제
Y_test_inverse = scaler_Y.inverse_transform(Y_test)  # 실제 값도 정규화 해제

# 시각화: 실제 값과 예측 값 비교
plt.figure(figsize=(10,6))
plt.plot(Y_test_inverse, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()