import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

# 20차 모델 성능을 알아보자능
np.random.seed(42)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

train_df = df.loc[:19]
train_df

for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i
    
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y = train_df["y"]

from sklearn.linear_model import Lasso

val_result=np.repeat(0, 100)
tr_result=np.repeat(0, 100)

for i in np.arange(0, 10, 0.1):
    model= Lasso(alpha=i*(0.1))
    model.fit(train_x, train_y)

    # 모델 성능
    y_hat_train = model.predict(train_x)
    y_hat_val = model.predict(valid_x)

    perf_train=sum((train_df["y"] - y_hat_train)**2)
    perf_val=sum((valid_df["y"] - y_hat_val)**2)
    tr_result[i]=perf_train
    val_result[i]=perf_val

# model= Lasso(alpha=0.1) # lamdea가 alpha로 표현
# 람다(λ)가 커질수록 베타 계수들이 0에 가까워지는 경향
# model.fit(train_x, train_y)
# hyperparameter는 validation이랑 train set보고 나중에 계산
# train data에서 모델을 학습시키고 valid data에서 모델의 성능을 평가한 후 최적의 hyperparameter 찾음

valid_df = df.loc[20:]
valid_df

for i in range(2, 21):
    valid_df[f"x{i}"] = valid_df["x"] ** i

# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]


# 모델 성능
y_hat_train = model.predict(train_x)
y_hat_val = model.predict(valid_x)

sum((train_df["y"] - y_hat_train)**2)
sum((valid_df["y"] - y_hat_val)**2)