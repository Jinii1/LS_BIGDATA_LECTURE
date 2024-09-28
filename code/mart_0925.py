import pandas as pd
import numpy as np

train=pd.read_csv('../data/mart_train.csv')
test=pd.read_csv('../data/mart_test.csv')

# mart 판매 데이터를 기반으로 판매액을 예측하시오

# 예측할 컬럼: total(총 판매액) 학습용 데이터(mart_train.csv)를 이용하여
# 총 판매액을 예측하는 모델을 만든 후,
# 이를 평가용 데이터에 적용하여 얻은 예측값을 다음과 같은 형식의 CSV 파일로 생성하시오

# pred: 예측된 총 판매액
# RMSE 평가지표에 따라 채점한다

# 문제가 회귀인지 분류인지

# 우선 변수를 빼지 마라 (빅분기는 도메인 지식 요구 x)
# 이상치도 빼지 마라

## 3. EDA 데이터 모양 파악
train.info()
test.info()

train.head()
test.head()

train.shape
test.shape # total 변수가 test에는 없음

target=train.pop('total') # drop이랑 같다, pop은 떨어트리면서 바로 저장이 됨
target.shape

train.info()
test.info()

## 4. Preprocessing: scaler, encoder 적용(minmax, robust 중 1개, labelencoder, onehotencoding 중 1개)
# 라벨 인코딩:
# 포도 1 사과 2 딸기 3
# 원핫 인코딩:
# 포도 100 사과 010 딸기 001

# rating이 숫자형 변수라서
# MinMaxScaler 쓸 수도 있고
cols=['rating']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train[cols]=scaler.fit_transform(train[cols])
test[cols]=scaler.fit_transform(test[cols])

# # RobustScaler 쓸 수도 있음
# from sklearn.preprocessing import RobustScaler
# scaler = RobustScaler()
# train[cols]=scaler.fit_transform(train[cols])
# test[cols]=scaler.fit_transform(test[cols])

train.head()

# 범주형 dummies처리
print(train.shape, test.shape) #(700,9) , (300,9)
train = pd.get_dummies(train)
test = pd.get_dummies(test)
print(train.shape, test.shape) #(700,30), (300,30)

## 5. Data Split 검증 데이터 분리: 무조건 암기
# random state 매개변수는 train_test_split함수에서 데이터 분할 시 사용되는 난수 생성기의 시드를 설정하는 역할
# 다른 사람과 동일한 데이터 분할을 사용하거나, 여러번 코드를 실행할 때 동일한 결과를 얻고 싶을 때 유용
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(train, target, test_size=0.2, random_state=2024)
print(X_tr.shape, X_val.shape, y_tr.shape, y_val.shape)

# ---------------------------------------------------------------
## 6. Model Learnign & Evaluation
# - RandomForestClassifier, RandomForestRegressor
# - predict, predict_proba(확률을 구할 때, ex. 동아리 가입 확률)

## 랜덤 포레스트
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_tr, y_tr)
pred = model.predict(X_val)
pred.shape # (140,)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(pred, y_val)**0.5)

## 7. Model Prediction
pred = model.predict(test)
submit = pd.DataFrame ({'pred': pred})
submit.to_csv('../data/submit.csv', index=False)
submit.shape

# ---------------------------------------------------------------

## 하이퍼파라미터 튜닝 (GridSearchCV)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# 하이퍼파라미터 그리드 설정
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}

# GridSearchCV 사용하여 최적의 모델 찾기
grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=5)
grid_search.fit(X_tr, y_tr)

# 최적 모델 찾기
best_model = grid_search.best_estimator_

# 검증 데이터를 사용하여 예측
pred = best_model.predict(X_val)

# RMSE 계산
from sklearn.metrics import mean_squared_error
print(mean_squared_error(pred, y_val)**0.5) # 413457.9798258445
# -----------------------------------------------------------------------
import lightgbm as lgb

# LightGBM 모델 하이퍼파라미터 그리드 설정
param_grid = {
    'num_leaves': [31, 50, 70, 100],
    'max_depth': [-1, 10, 20, 30],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 500, 1000],
    'min_child_samples': [10, 20, 30],
    'subsample': [0.6, 0.8, 1.0]
}

# LightGBM 모델 생성 및 GridSearchCV 사용
lgb_model = lgb.LGBMRegressor(random_state=2024)

grid_search = GridSearchCV(estimator=lgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_tr, y_tr)

# 최적 모델 출력
best_lgb_model = grid_search.best_estimator_

# 검증 데이터를 사용하여 예측
pred = best_lgb_model.predict(X_val)

# RMSE 계산
rmse = mean_squared_error(pred, y_val)**0.5
print("RMSE:", rmse) # RMSE: 396560.7840716658
# ------------------------------------------------------------------------
# 앙상블 모델 설정 (베이스 모델 + 메타 모델)
# ! pip install xgboost
from xgboost import XGBRegressor
base_models = [
    ('random_forest', RandomForestRegressor(random_state=2024)),
    ('lightgbm', lgb.LGBMRegressor(random_state=2024)),
    ('xgboost', XGBRegressor(random_state=2024))
]

# 메타 모델로 Ridge 사용
stacking_model = StackingRegressor(estimators=base_models, final_estimator=Ridge())

# 앙상블 모델 학습
stacking_model.fit(X_tr, y_tr)

# 검증 데이터로 예측 수행
pred = stacking_model.predict(X_val)

# RMSE 계산
rmse = mean_squared_error