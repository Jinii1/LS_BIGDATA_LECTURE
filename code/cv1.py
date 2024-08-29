import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

# 20차 모델 성능을 알아보자
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8) # -4부터 4까지
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3) # 약간의 노이즈

df = pd.DataFrame({
    "y" : y,
    "x" : x
})

# x 제곱, 세제곱 ... 등의 변환을 2부터 20까지 반복
for i in range(2, 21):
    df[f"x{i}"] = df["x"] ** i

df

# 훈련과 검증 데이터셋 나누기
def make_tr_val(fold_num, df):
    np.random.seed(2024)
    myindex=np.random.choice(30, 30, replace=False) # 0부터 29까지 무작위로 순서 섞어서 뽑기

    # valid index
    val_index=myindex[(10*fold_num):(10*fold_num+10)] # 30개 중 일부를 검증용으로 선택

    # valid set, train set
    valid_set=df.loc[val_index] # 검증용 데이터
    train_set=df.drop(val_index) # 훈련용 데이터

    # 독립변수와 종속변수로 나눠서 반환
    train_X=train_set.iloc[:,1:]
    train_y=train_set.iloc[:,0]

    valid_X=valid_set.iloc[:,1:]
    valid_y=valid_set.iloc[:,0]

    return (train_X, train_y, valid_X, valid_y)

# 라쏘: overfitting 막아주는 알고리즘
# Overfitting: 모델이 훈련 데이터에 너무 맞춰져서 새로운 데이터를 잘 예측하지 못하는 현상
from sklearn.linear_model import Lasso

val_result_total=np.repeat(0.0, 3000).reshape(3, -1)
tr_result_total=np.repeat(0.0, 3000).reshape(3, -1)

for j in np.arange(0, 3):
    train_X, train_y, valid_X, valid_y = make_tr_val(fold_num=j, df=df)

    # 결과 받기 위한 벡터 만들기
    val_result=np.repeat(0.0, 1000)
    tr_result=np.repeat(0.0, 1000)

    for i in np.arange(0, 1000):
        model= Lasso(alpha=i*0.01)
        model.fit(train_X, train_y)

        # 모델 성능
        y_hat_train = model.predict(train_X)
        y_hat_val = model.predict(valid_X)

        # 훈련용과 검증용 데이터에서의 예측 성능 계산
        perf_train=sum((train_y - y_hat_train)**2)
        perf_val=sum((valid_y - y_hat_val)**2)
        tr_result[i]=perf_train
        val_result[i]=perf_val

    tr_result_total[j,:]=tr_result
    val_result_total[j,:]=val_result


import seaborn as sns

df = pd.DataFrame({
    'lambda': np.arange(0, 10, 0.01), 
    'tr': tr_result_total.mean(axis=0),
    'val': val_result_total.mean(axis=0)
})

df['tr']

# seaborn을 사용하여 산점도 그리기
# lambda값에 따라 모델 성능이 어떻게 변하는지 확인
sns.scatterplot(data=df, x='lambda', y='tr')
sns.scatterplot(data=df, x='lambda', y='val', color='red')
plt.xlim(0, 10)
plt.ylim(0, 0.4)

val_result[0]
val_result[1]
np.min(val_result_total.mean(axis=0))

# alpha를 2.67로 선택!
np.argmin(val_result_total.mean(axis=0))
# 성능이 제일 좋은 alpha값 찾기
# val_result_total.mean(axis=0) 값이 제일 작을때(예측이 제일 잘 맞을 때)의 lambda값 찾기
np.arange(0, 10, 0.01)[np.argmin(val_result_total.mean(axis=0))]