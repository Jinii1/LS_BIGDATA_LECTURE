import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

df=penguins.dropna()
df.isna().sum()
df=df[['bill_length_mm', 'bill_depth_mm']]
df=df.rename(columns={'bill_length_mm': 'y',
                      'bill_depth_mm': 'x'})
df

# 원래 MSE는?
np.mean((df["y"] - df["y"].mean())**2) # 29.81

# x=51 기준으로 나눴을 떄, 데이터 포인트가 몇개씩 나뉘나요?
# df에서 x값이 15을 기준으로 나눴을 떄, 그 기준에 따라 데이터가 몇 개씩 나뉘는지 확인
# df를 특정 기준으로 분리하여 각 그룹에 속하는 데이터 포인트 수 확인하는 작업 요구 
n1=df.query("x < 15").shape[0] # 1번 그룹: 276
n2=df.query("x >= 15").shape[0] # 2번 그룹: 57

# 1번 그룹은 얼마로 예측하나요? 각 그룹에 대한 예측값 추정
# 2번 그룹은 얼마로 예측하나요?
y_hat1=df.query('x<15').mean()[0]
y_hat2=df.query('x>=15').mean()[0]

# 각 그룹 MSE는 얼마인가요?
mse1=np.mean((df.query("x < 15")['y'] - y_hat1)**2)
mse2=np.mean((df.query('x>=15')['y'] - y_hat2)**2)

# x=15의 MSE 가중평균은?
# (mse1+mse2)*0.5가 아닌
(mse1*n1 + mse2*n2)/(n1+n2) # 29.23

# 원래 MSE와 x=15의 MSE S가중평균 차이
29.81 - 29.23
# ==========================================================
# x=20의 MSE 가중평균은?
n1=df.query('x<20').shape[0] # 1번 그룹: 311
n2=df.query('x>=20').shape[0] # 2번 그룹: 22

y_hat1=df.query("x < 20").mean()[0]
y_hat2=df.query("x >= 20").mean()[0]

mse1=np.mean((df.query("x < 20")["y"] - y_hat1)**2)
mse2=np.mean((df.query("x >= 20")["y"] - y_hat2)**2)

(mse1* n1 + mse2 * n2)/(n1+n2)
29.73

29.81-29.73