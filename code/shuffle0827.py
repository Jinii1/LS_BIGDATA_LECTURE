# train set을 5개로 쪼개어 valid set과 train set을 5개로 만들기
# 각 세트에 대한 성능을 각 lambda값에 대하여 구하기
# 성능평가지표 5개를 평균내어 오른쪽 그래프 다시 그리기

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import Lasso
import pandas as pd


np.random.seed(2024)

x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)
df = pd.DataFrame({
    "y" : y,
    "x" : x
})

for i in range(2, 21):
    df[f"x{i}"] = df["x"] ** i

df

myindex=np.random.choice(30, 30, replace=False)

myindex[0:10]
myindex[10:20]
myindex[20:30]

from sklearn.linear_model import Lasso

# 결과 받기 위한 벡터 만들기
val_result=np.repeat(0.0, 100)
tr_result=np.repeat(0.0, 100)

for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_X, train_y)

    # 모델 성능
    y_hat_train = model.predict(train_X)
    y_hat_val = model.predict(valid_X)

    perf_train=sum((train_y - y_hat_train)**2)
    perf_val=sum((valid_y - y_hat_val)**2)
    tr_result[i]=perf_train
    val_result[i]=perf_val

tr_result
val_result