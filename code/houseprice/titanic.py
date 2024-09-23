# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder

# 워킹 디렉토리 설정
import os
cwd = os.getcwd()

# 필요한 데이터 불러오기
train_raw = pd.read_csv("../../data/titanic/train.csv")
test_raw = pd.read_csv("../../data/titanic/test.csv")
sub_df_raw = pd.read_csv("../../data/titanic/sample_submission.csv")

# 데이터 복사본 만들기
train = train_raw.copy()
test = test_raw.copy()
sub_df = sub_df_raw.copy()

# NaN 채우기
# 각 숫자형 변수는 평균으로, 범주형 변수는 최빈값(Mode)으로 채우기

## Train 데이터
# 숫자형 채우기
quantitative = train.select_dtypes(include=[int, float])
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]
for col in quant_selected:
    train[col].fillna(train[col].mean(), inplace=True)

# 범주형 채우기
qualitative = train.select_dtypes(include=[object])
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]
for col in qual_selected:
    train[col].fillna(train[col].mode()[0], inplace=True)

## Test 데이터
# 숫자형 채우기
quantitative = test.select_dtypes(include=[int, float])
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]
for col in quant_selected:
    test[col].fillna(train[col].mean(), inplace=True)

# 범주형 채우기
qualitative = test.select_dtypes(include=[object])
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]
for col in qual_selected:
    test[col].fillna(train[col].mode()[0], inplace=True)

# 'Transported' 변수 제외한 후 데이터 통합
df_train = train.drop('Transported', axis=1)
df_test = test.copy()

# 'PassengerId' 제거 후 더미 코딩 (범주형, bool형 변수를 대상으로)
df_train = pd.get_dummies(df_train.drop('PassengerId', axis=1), drop_first=True)
df_test = pd.get_dummies(df_test.drop('PassengerId', axis=1), drop_first=True)

# train 데이터와 test 데이터 간의 컬럼이 동일해야 함
df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)

# 종속변수와 독립변수 나누기
train_x = df_train  # 'Transported'를 제외한 훈련 데이터
train_y = train['Transported']  # 'Transported' 종속 변수

test_x = df_test.copy()  # test 데이터

# 데이터 스케일링 (StandardScaler 사용)
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

# 결과 확인
print(f"Train shape: {train_x_scaled.shape}, Test shape: {test_x_scaled.shape}")
