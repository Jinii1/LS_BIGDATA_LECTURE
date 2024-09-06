# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

# 워킹 디렉토리 설정
import os
cwd=os.getcwd()

## 필요한 데이터 불러오기
house_train=pd.read_csv("../../data/houseprice/train.csv")
house_test=pd.read_csv("../../data/houseprice/test.csv")
sub_df=pd.read_csv("../../data/houseprice/sample_submission.csv")

## NaN 채우기
# 각 숫치형 변수는 평균 채우기
# 각 범주형 변수는 Unknown 채우기
house_train.isna().sum()
house_test.isna().sum()

## 숫자형 채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)
house_train[qual_selected].isna().sum()

house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
# house_test.select_dtypes(include=[int, float])

df = pd.concat([house_train, house_test], ignore_index=True)
# df.info()
df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

# Validation 셋(모의고사 셋) 만들기
np.random.seed(42)
val_index=np.random.choice(np.arange(train_n), size=438, replace=False)
val_index

# train => valid / train 데이터셋
valid_df=train_df.loc[val_index]  # 30%
train_df=train_df.drop(val_index) # 70%

## 이상치 탐색
train_df=train_df.query("GrLivArea <= 4500")

# x, y 나누기
# regex (Regular Expression, 정규방정식)
# ^ 시작을 의미, $ 끝남을 의미, | or를 의미
# selected_columns=train_df.filter(regex='^GrLivArea$|^GarageArea$|^Neighborhood_').columns

## train
train_x=train_df.drop("SalePrice", axis=1)
train_y=train_df["SalePrice"]

## valid
valid_x=valid_df.drop("SalePrice", axis=1)
valid_y=valid_df["SalePrice"]

## test
test_x=test_df.drop("SalePrice", axis=1)

# 선형 회귀 모델 생성
model = LinearRegression()

# Lasso 모델 생성
lasso_model = Lasso(max_iter=500)
param_grid={
    'alpha': [0.1, 1.0, 10.0, 100.0]
}

grid_search=GridSearchCV( # 각 그리드별로 성능평가
    estimator=lasso_model, # 평가할 모델
    param_grid=param_grid, # parameter 설정할 수 있는 후보군
    scoring='neg_mean_squared_error', # 성능평가할땐 mean squared 사용해라
    cv=5 # train set을 5개로 쪼개서 한 그룹 valid x 5
)

# Ridge 모델 생성
model = Ridge(alpha=0.03)

# ==============================================================
# ElasticNet 모델 생성
# -> 최적의 hyperparameter 찾기 위한 grid search 실행하고 최적의 모델을 사용하여 예측 수행

model=ElasticNet() # 원하는 모델 선언
param_grid={
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0, 0.1, 0.5, 1.0]
}
# param_grid: 그리드 서치에서 테스트할 하이퍼파라미터 후보군
# alpha: 규제 강도
# l1_ratio: 라쏘(1)와 릿지(0) 사이의 비율 조정
# -> 다양한 조합의 규제 강도와 규제 비율 실험해서 가장 적합한 조합 찾기 위함

grid_search=GridSearchCV( # 각 그리드별로 성능평가
    estimator=model, # 평가할 ElasticNet 모델
    param_grid=param_grid, # parameter 후보군, alpha와 l1_ratio 조합 시도
    scoring='neg_mean_squared_error', # 성능평가할땐 mean squared 사용해라
# 성능평가기준: MSE 음수값 (음수인 이유: MSE가 작은 것이 더 좋은 성능이기에)
    cv=5 # 5겹 교차검증, train set을 5개로 쪼개서 한 그룹 valid x 5
)
grid_search.fit(train_x, train_y)
# 그리드 내 모든 parameter 조합 시도한 후, 가장 성능이 좋은 hyper parameter 선택

grid_search.best_params_ # grid search에서 찾은 최적의 hyper parameter 값 반환
grid_search.cv_results_ # 모든 hyper parameter 조합의 성능평가결과 변환
# -> 각 조합에 대해 교차 검증에서 얻은 MSE 값, 실행시간 포함
grid_search.best_score_ # 최고 성능
best_model=grid_search.best_estimator_ # grid search에서 찾은 최적의 hyper parameter를 가진 최적의 ElasticNet 모델 반환
# -> 이 모델을 통해 새로운 데이터 예측 가능

best_model.predict(valid_x) # predict 함수 사용가능
# ==========================================================
# 모델 학습
model.fit(train_x, train_y)  # 자동으로 기울기, 절편 값을 구해줌

# 성능 측정 ()
y_hat=model.predict(valid_x)
np.sqrt(np.mean((valid_y-y_hat)**2))

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission10.csv", index=False)