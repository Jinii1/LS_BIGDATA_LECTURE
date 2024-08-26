# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 워킹 디렉토리 설정
import os
cwd=os.getcwd()

## 필요한 데이터 불러오기
house_train=pd.read_csv("../../data/houseprice/train.csv")
house_test=pd.read_csv("../../data/houseprice/test.csv")
sub_df=pd.read_csv("../../data/houseprice/sample_submission.csv")

## 이상치 탐색 (여기 넣으면 안됨!)
# house_train=house_train.query("GrLivArea <= 4500")

house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
df = pd.concat([house_train, house_test], ignore_index=True)
# 인덱스를 무시하고 새로 부여된 연속적인 인덱스를 사용
df = pd.get_dummies(
    df,
    columns= ["Neighborhood"],
    drop_first=True
    )
# 변환된 더미 변수 중에서 첫 번째 범주를 삭제
df

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

# Validation 셋(모의고사 셋) 만들기
np.random.seed(42)
val_index=np.random.choice(np.arange(train_n), size=438, replace=False)
# 복원 추출 x
val_index

# train => valid / train 데이터셋
valid_df=train_df.loc[val_index]  # 30%
train_df=train_df.drop(val_index) # 70%

## 이상치 탐색
train_df=train_df.query("GrLivArea <= 4500")

# x, y 나누기
# filter는 특정 조건에 맞는 열(또는 행)을 선택
# regex (Regular Expression, 정규방정식)
# ^ 시작을 의미, $ 끝남을 의미, | or를 의미
# Neighborhood_로 표시함으로써 Neighborhood 변수 다 불러옴
selected_columns=train_df.filter(regex='^GrLivArea$|^GarageArea$|^Neighborhood_').columns

## train
train_x=train_df[selected_columns]
train_y=train_df["SalePrice"]

## valid
valid_x=valid_df[selected_columns]
valid_y=valid_df["SalePrice"]

## test
test_x=test_df[selected_columns]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(train_x, train_y)  # 자동으로 기울기, 절편 값을 구해줌

# 성능 측정 ()
y_hat=model.predict(valid_x)
np.sqrt(np.mean((valid_y-y_hat)**2))

## test 셋 결측치 채우기
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()
test_x=test_x.fillna(house_test["GarageArea"].mean())

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission10.csv", index=False)