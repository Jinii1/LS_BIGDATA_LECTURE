# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import os
cwd=os.getcwd()

## 필요한 데이터 불러오기
house_train=pd.read_csv("data/houseprice/train.csv")
house_test=pd.read_csv("data/houseprice/test.csv")
sub_df=pd.read_csv("data/houseprice/sample_submission.csv")

# 이상치 탐색 (이 부분은 여기에 넣으면 안 됨)
# house_train=house_train.query("GrLivArea <= 4500")

# dummies를 한 번에 구하고 나누려고?
house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기
df=pd.concat([house_train, house_test],ignore_index=True)
df

# 통합 df 만들기 + 더미코딩
df = pd.get_dummies(
    df,
    columns= ['Neighborhood'],
    drop_first=True
    )
df

# train/ test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

# validation셋 (모의고사 셋) 만들기
np.random.seed(42)
val_index=np.random.choice(np.arange(train_n), size=438, replace=False)
val_index

# train => valid/ train 데이터셋
valid_df=train_df.loc[val_index]    # 30%
train_df=train_df.drop(val_index)   # 70%

# 이상치 탐색
train_df=train_df.query('GrLivArea<=4500')

# x, y 나누기
# regex=정규방정식
# ^ 시작을 의미,$ 끝남을 의미, | or을 의미
selected_columns = train_df.filter(regex='^GrLivArea$|^GarageArea$|^Neighborhood_').columns
train_x=train_df[selected_columns]
train_y=train_df['SalePrice']

valid_x=train_df[selected_columns]
valid_y=valid_df['SalePrice']

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(train_x, train_y)  # 자동으로 기울기, 절편 값을 구해줌

# 성능 측정
y_hat=model.predict(valid_x)
np.sqrt(np.mean((valid_y-y_hat)**2))


# -> 이걸 한 이유?
# 1. 어떤 변수를 사용하는 모델이 더 좋은지
# 2. 내가 제외한 이상치가 진짜 성능을 높이는지 낮추는지 알 수 있다

# 결과가 성능 측정이 낮을수록 좋다 왜냐면 값이 오차니까