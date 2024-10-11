import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 데이터 불러오기
df= pd.read_csv('../../bigfile/week2_data.csv', encoding='cp949')
df_asos = pd.read_csv("../../bigfile/OBS_ASOS_TIM_20200601-01_20200824-23.csv", encoding='euc-kr')

df

# 열 이름에서 단위 제거
df.columns = [col.split('(')[0].strip() for col in df.columns]
df_asos.columns = [col.split('(')[0].strip() for col in df_asos.columns]

locations_to_drop = ['강화', '거제', '거창', '고흥', '구미', '군산', '금산', '남원', '남해', '동두천', '동해', 
                     '문경', '밀양', '보령', '보은', '봉화', '부안', '부여', '산청', '상주', '서귀포', '성산', 
                     '세종', '속초', '순천', '양평', '영덕', '영월', '영주', '영천', '완도', '울산', '울진', 
                     '의성', '이천', '인제', '임실', '장수', '장흥', '정선군', '정읍', '제천', '진도군', 
                     '천안', '충주', '태백', '통영', '파주', '합천', '해남', '홍천']

# '지점명'이 해당 목록에 포함되지 않은 행들만 남기기 (삭제)
df_asos = df_asos[~df_asos['지점명'].isin(locations_to_drop)]
df_asos.isna().sum()/len(df_asos)

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
df['day_of_year'] = df['date_time'].dt.dayofyear

df_asos['month'] = df_asos['일시'].dt.month
df_asos['day'] = df_asos['일시'].dt.day
df_asos['hour'] = df_asos['일시'].dt.hour
df_asos['day_of_year'] = df_asos['일시'].dt.dayofyear

# -------------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

sns.lineplot(data=df_asos.query('지점명=="청주"'), x='day_of_year',y='일사')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우의 경우
# 마이너스 폰트 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt

# 지점별 일사량을 박스플롯으로 시각화
plt.figure(figsize=(15, 8))
df_asos.boxplot(column='일사', by='지점명', grid=False, vert=True, patch_artist=True, figsize=(20, 8))

# 그래프 설정
plt.title('지역별 일사량 분포 비교', fontsize=16)
plt.suptitle('')  # 기본제목 제거
plt.xlabel('지점명', fontsize=12)
plt.ylabel('일사량 (MJ/m²)', fontsize=12)
plt.xticks(rotation=90)  # 지점명이 잘 보이도록 회전
plt.tight_layout()

# 그래프 출력
plt.show()
# -------------------------------------------------------------------------
