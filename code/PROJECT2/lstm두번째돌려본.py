import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# 1. 데이터 로드
train = pd.read_csv('../../bigfile/train_chungju.csv')
test = pd.read_csv('../../bigfile/test_chungju.csv')

# 2. 전처리: 컬럼 이름 수정 및 결측값 처리
train.columns = [col.split('(')[0].strip() for col in train.columns]
test.columns = [col.split('(')[0].strip() for col in test.columns]

# 날짜 데이터를 datetime 형식으로 변환
train['일시'] = pd.to_datetime(train['일시'])
test['일시'] = pd.to_datetime(test['일시'])

# 필요한 파생 변수 생성 (month, day, hour)
for df in [train, test]:
    df['month'] = df['일시'].dt.month
    df['day'] = df['일시'].dt.day
    df['hour'] = df['일시'].dt.hour

# 결측값 처리 (0으로 대체)
for col in ['일조', '일사', '강수량', '풍속']:
    train[col] = train[col].fillna(0.0)
    test[col] = test[col].fillna(0.0)

# 3. X 변수와 Y 변수 설정
X_columns = ['기온', '강수량', '풍속', '습도', '일조', 'month', 'day', 'hour']
Y_column = ['일사']

# 4. 데이터 스케일링 (0~1 범위로 정규화)
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))

# 학습 데이터 정규화
X_train_scaled = scaler_X.fit_transform(train[X_columns])
Y_train_scaled = scaler_Y.fit_transform(train[Y_column])

# 테스트 데이터 정규화
X_test_scaled = scaler_X.transform(test[X_columns])
Y_test_scaled = scaler_Y.transform(test[Y_column])

# 5. LSTM 모델을 위한 데이터셋 생성 함수
def create_sequences(X, Y, time_steps=60):
    X_seq, Y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])  # time_steps 길이만큼의 시퀀스 생성
        Y_seq.append(Y[i + time_steps])    # 해당 시퀀스의 다음 값을 예측
    return np.array(X_seq), np.array(Y_seq)

# 시퀀스 생성
time_steps = 60
X_train_seq, Y_train_seq = create_sequences(X_train_scaled, Y_train_scaled, time_steps)
X_test_seq, Y_test_seq = create_sequences(X_test_scaled, Y_test_scaled, time_steps)

# 6. LSTM 모델 구성
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# 7. 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 8. 모델 학습
history = model.fit(X_train_seq, Y_train_seq, epochs=10, batch_size=32, validation_data=(X_test_seq, Y_test_seq))

# 9. 예측 수행
train_predict = model.predict(X_train_seq)
test_predict = model.predict(X_test_seq)

# 10. 예측값과 실제값을 스케일링 해제
train_predict = scaler_Y.inverse_transform(train_predict)
test_predict = scaler_Y.inverse_transform(test_predict)
Y_train_seq_inv = scaler_Y.inverse_transform(Y_train_seq)
Y_test_seq_inv = scaler_Y.inverse_transform(Y_test_seq)

# 11. 시각화: 실제 값과 예측 값 비교 (테스트 데이터)
plt.figure(figsize=(14, 5))
plt.plot(Y_test_seq_inv, label='Actual')
plt.plot(test_predict, label='Predicted')
plt.title('실제 값과 예측 값 비교 (테스트 데이터)')
plt.xlabel('Time Steps')
plt.ylabel('일사')
plt.legend()
plt.show()

# 학습 손실 그래프
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Train and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 10. 성능 평가: MAE (Mean Absolute Error)
train_mae = mean_absolute_error(Y_train_seq_inv, train_predict)
test_mae = mean_absolute_error(Y_test_seq_inv, test_predict)

print(f'Train MAE: {train_mae}')
print(f'Test MAE: {test_mae}')