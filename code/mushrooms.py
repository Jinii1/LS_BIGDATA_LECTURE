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
cwd=os.getcwd()

## 필요한 데이터 불러오기
mush_train_raw = pd.read_csv("../data/mushroom/train.csv")
mush_test_raw = pd.read_csv("../data/mushroom/test.csv")
sub_df_raw = pd.read_csv("../data/mushroom/sample_submission.csv")

mush_train_raw.info() # cap-diameter, stem-height, stem-width만 float/ 나머지 object
mush_train_raw.shape # (3116945, 22)

# e: 식용 p: 독성

os.getcwd()

mush_train = mush_train_raw.copy()
mush_test = mush_test_raw.copy()
sub_df = sub_df_raw.copy()

mush_train.isna().sum()
mush_test.isna().sum()

columns_10_7 = mush_train.columns[mush_train.isna().mean() * 100>=60]
drop_columns = ['id'] + list(columns_10_7)

mush_train.drop(columns = drop_columns,inplace=True)
mush_test.drop(columns = drop_columns,inplace=True)

mush_train.isna().sum()
mush_test.isna().sum()

categorical_columns = mush_train.select_dtypes(include=['object']).columns
numerical_columns = mush_train.select_dtypes(include=[int, float]).columns
categorical_columns_test = mush_test.select_dtypes(include=['object']).columns
numerical_columns_test = mush_test.select_dtypes(include=[int, float]).columns
#df = pd.concat([mush_train,mush_test])

for column in categorical_columns:
    label_encoder = LabelEncoder()
    mush_train[column] = label_encoder.fit_transform(mush_train[column])
mush_train

for column in categorical_columns_test:
    label_encoder = LabelEncoder()
    mush_test[column] = label_encoder.fit_transform(mush_test[column])

def fillna_mean(df):
    target_columns = df.columns[df.isna().sum() != 0]
    for col in target_columns:
        df[col] = df[col].fillna(df[col].mean())

fillna_mean(mush_train)
fillna_mean(mush_test)
