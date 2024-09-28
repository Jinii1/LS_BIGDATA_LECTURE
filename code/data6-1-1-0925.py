# 출동시간과 도착시간 차이가 평균적으로 가장 오래 걸린 소방서의 시간을 분으로 변환해 출력하시오
# (반올림 후 정수 출력)

import pandas as pd
df=pd.read_csv('../data/data6-1-1.csv')

df['출동시간'] = pd.to_datetime(df['출동시간'])
df['도착시간'] = pd.to_datetime(df['도착시간'])
df.info()

df['차이'] = (df['도착시간'] - df['출동시간']).dt.total_seconds()/60
df=df.groupby('소방서')['차이'].mean()
df=df.sort_values(ascending=False)
df
int(round(df[0]))