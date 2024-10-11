import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 데이터 불러오기
raw_data = pd.read_csv('../../bigfile/week2_data.csv', encoding='cp949')
df = raw_data.copy()

df_asos = pd.read_csv("data/OBS_ASOS_TIM.csv", encoding='euc-kr')

# 열 이름에서 단위 제거
df.columns = [col.split('(')[0].strip() for col in df.columns]
df_asos.columns = [col.split('(')[0].strip() for col in df_asos.columns]

# 결측치 처리
df_asos['일조'] = df_asos['일조'].fillna(0.0)
df_asos['일사'] = df_asos['일사'].fillna(0.0)
df_asos['강수량'] = df_asos['강수량'].fillna(0.0)
df_asos['기온'] = df_asos['기온'].fillna(df_asos['기온'].mean())
df_asos['풍속'] = df_asos['풍속'].fillna(df_asos['풍속'].mean())
df_asos['습도'] = df_asos['습도'].fillna(df_asos['습도'].mean())

# 날짜와 시간 처리
df['date_time'] = pd.to_datetime(df['date_time'])
df_asos['일시'] = pd.to_datetime(df_asos['일시'])

# month/day를 month와 day로 분리하고 hour를 추가
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.day
df['hour'] = df['date_time'].dt.hour

df_asos['month'] = df_asos['일시'].dt.month
df_asos['day'] = df_asos['일시'].dt.day
df_asos['hour'] = df_asos['일시'].dt.hour

# 사용하지 않는 열 제거
df = df.drop(['num', 'date_time', '전력사용량', '비전기냉방설비운영', '태양광보유'], axis=1)
df_asos = df_asos.drop(['일시', '지점', '지점명'], axis=1)

# 열 순서 맞추기
df = df[['month', 'day', 'hour', '기온', '강수량', '풍속', '습도', '일조']]
df_asos = df_asos[['month', 'day', 'hour', '기온', '강수량', '풍속', '습도', '일조', '일사']]
