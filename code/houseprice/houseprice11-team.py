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

# train으로 mean값으로 해야 하는 것
# trainset만 아는 상태에서만 예측하려고 햇우니까

# test 데이터 채우기
## 숫자형 채우기
quantitative = house_test.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_test[col].fillna(house_train[col].mean(), inplace=True)
house_test[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_test.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_test[col].fillna("unknown", inplace=True)
house_test[qual_selected].isna().sum()


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

## 이상치 탐색
train_df=train_df.query("GrLivArea <= 4500")

## train
train_x=train_df.drop("SalePrice", axis=1)
train_y=train_df["SalePrice"]

## test
test_x=test_df.drop("SalePrice", axis=1)

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

eln_model= ElasticNet()
rf_model= RandomForestRegressor(n_estimators=150) # 트리 개수(부트스트랩이여기에담겨있다?)

np.random.seed(20240911)
# 그리드 서치 for ElasticNet
param_grid = {
    'alpha': np.arange(64, 65, 0.1), 
    'l1_ratio': [0, 0.1, 0.5, 1.0]
}
grid_search=GridSearchCV(
    estimator=eln_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error', # 평가기준: MSE
    cv=5 # 5-fold 교차 검증
)
grid_search.fit(train_x, train_y)
best_eln_model=grid_search.best_estimator_

# 그리드 서치 for RandomForests

# RandomForest model: 다수의 Decision Tree기반으로 앙상블 학습 수행 모델
# 각각의 트리들이 독립적으로 학습한 후, 이들의 예측을 종합하여 최종 예측 수행

param_grid = {
    'max_depth': [3, 5, 7],                  # 트리의 최대 깊이
    'min_samples_split': [20, 10, 5],        # 노드를 분할하는 데 필요한 최소 샘플 수
    'min_samples_leaf': [5, 10, 20, 30],     # 리프 노드(말단 노드)에 필요한 최소 샘플 수
    'max_features': ['sqrt', 'log2', None] # None: 전체 갯수 고려, log2는 그냥 수를 줄이는 것 # 각 노드에서 사용할 최대 특성의 수
}

grid_search=GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search.fit(train_x, train_y)
grid_search.best_params_
best_rf_model=grid_search.best_estimator_

# Stacking
y1_hat=best_eln_model.predict(train_x) # test 셋에 대한 집값
y2_hat=best_rf_model.predict(train_x) # test 셋에 대한 집값

# 각 모델의 예측값을 새로운 학습 데이터로 사용
train_x_stack=pd.DataFrame({
    'y1': y1_hat,
    'y2': y2_hat
})

# Lasso 모델을 블렌더로 사용하여 학습
from sklearn.linear_model import Lasso
ls_model = Lasso()

param_grid={
    'alpha': np.arange(0, 10, 0.01)
}

grid_search=GridSearchCV(
    estimator=ls_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search.fit(train_x_stack, train_y)
grid_search.best_params_
blander_model=grid_search.best_estimator_

blander_model.coef_
# 엘라스틱넷 예측값에 0.36한 값과 랜덤포레스트 예측값의 0.69
blander_model.intercept_

pred_y_eln=best_eln_model.predict(test_x) # test 셋에 대한 집값
pred_y_rf=best_rf_model.predict(test_x) # test 셋에 대한 집값

test_x_stack=pd.DataFrame({
    'y1': pred_y_eln,
    'y2': pred_y_rf
})

pred_y=blander_model.predict(test_x_stack)

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("../../data/houseprice/sample_submission10.csv", index=False)