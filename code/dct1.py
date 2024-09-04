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

# x=15 기준으로 나눴을 떄, 데이터 포인트가 몇개씩 나뉘나요?
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
# ==========================================================
## 4교시
# 기준값 x를 넣으면 MSE값이 나오는 함수는?
def my_mse(x) :
    n1 = df.query(f"x < {x}").shape[0] # 1번 그룹
    n2 = df.query(f"x >= {x}").shape[0] # 2번 그룹
    y_hat1 = df.query(f"x < {x}")["y"].mean()
    y_hat2 = df.query(f"x >= {x}")["y"].mean()
    mse1 = np.mean((df.query(f"x < {x}")["y"]-y_hat1)**2)
    mse2 = np.mean((df.query(f"x >= {x}")["y"]-y_hat2)**2)
    return (mse1 * n1 + mse2 * n2) / (n1 + n2)

my_mse(20)

df['x'].min()
df['x'].max()

# 13~22 사이 값 중 0.01 간격으로 MSE 계산을 해서
# minimize 사용해서 가장 작은 MSE가 나오는 x는?
# (=데이터를 가장 잘 나눌 수 있는 최적의 기준점 찾기)

# -> 데이터의 최적의 분리점(x)를 찾기 위해 MSE 최소화하는 방법 사용

x_values=np.arange(13.2, 21.4, 0.01)
len(x_values) # 820개
# x의 값을 13.2에서 21.4 사이의 0.01 간격으로 변화
# 각 x값에서의 MSE 계산해서 이 값을 result에 저장
result=np.repeat(0.0, 820)
for i in range(820):
    result[i]=my_mse(x_values[i])
result
x_values[np.argmin(result)] # 16.40
# 모든 x값에 대해 계산된 MSE 중 가장 작은 값 가지는 x
# x가 16.40일때 MSE가 가장 작아짐
# (=x가 16.40일 때 y의 예측값과 실제값 차이가 가장 적게되는 기준)
# ==========================================================
## 6교시
# 13~22 사이 값 중 0.01 간격으로 MSE 계산을 해서
# minimize 사용해서 가장 작은 MSE가 나오는 x 찾아보세요!
x_values=np.arange(16.51, 21.5, 0.01)
nk=x_values.shape[0]
result=np.repeat(0.0, nk)
for i in range(nk):
    result[i]=my_mse(x_values[i])

result
x_values[np.argmin(result)]
# 14.01, 16.42, 19.4

# x, y 산점도를 그리고, 빨간 평행선 4개 그려주세요!
import matplotlib.pyplot as plt

df.plot(kind="scatter", x="x", y="y")
thresholds=[14.01, 16.42, 19.4]
df["group"]=np.digitize(df["x"], thresholds)
y_mean=df.groupby("group").mean()["y"]
k1=np.linspace(13, 14.01, 100)
k2=np.linspace(14.01, 16.42, 100)
k3=np.linspace(16.42, 19.4, 100)
k4=np.linspace(19.4, 22, 100)
plt.plot(k1, np.repeat(y_mean[0],100), color="red")
plt.plot(k2, np.repeat(y_mean[1],100), color="red")
plt.plot(k3, np.repeat(y_mean[2],100), color="red")
plt.plot(k4, np.repeat(y_mean[3],100), color="red")