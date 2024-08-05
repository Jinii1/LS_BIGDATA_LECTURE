## 지난주 복습

# 필요한 패키지 불러오기
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd


# 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test =  pd.read_csv('./data/houseprice/test.csv')
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

house_train.info()

# 이상치 탐색 (그럼 이상치 두개만 빼진 house_train data 생성)
house_train=house_train.query('GrLivArea<=4500')

# 회귀분석 적합(fit)하기
# house_train['SalePrice'] # 판다스 시리즈
# house_train[['SalePrice']] #판다스 프레임
# 숫자형 변수만 선택하기
x = house_train.select_dtypes(include=[int, float])
x
x = house_train[['GrLivArea', 'GarageArea']]
y = np.array(house_train['SalePrice'])

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값 구해줌

# 회귀 직선의 기울기와 절편
model.coef_ # 기울기 = a값
model.intercept_ # 절편 = b값
slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 테스트 데이터 예측
test_x=house_test[['GrLivArea', 'GarageArea']]
test_x

# 결측치 확인
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()
# test_x=test_x.fillna(house_test["GarageArea"].mean(), inplace=True)
test_x=test_x.fillna(house_test["GarageArea"].mean())

# 테스트 데이터 집값 예측
pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission7.csv", index=False)
=======================================================================================
## 0805
# 모델의 성능을 높이기 위해서 어떻게 하면 좋을까?
# 숫자형 변수 다 넣어보기

# 필요한 패키지 불러오기
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd


# 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test =  pd.read_csv('./data/houseprice/test.csv')
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

house_train.info()

# 이상치 탐색 (그럼 이상치 두개만 빼진 house_train data 생성)
house_train=house_train.query('GrLivArea<=4500')

# 회귀분석 적합(fit)하기
# house_train['SalePrice'] # 판다스 시리즈
# house_train[['SalePrice']] #판다스 프레임
# 숫자형 변수만 선택하기
x = house_train.select_dtypes(include=[int, float])
x
# 필요없는 칼럼 제거하기
# (이렇게 되면 id도 학습이 돼서 id는 제외하려고 하고 마지막 변수 saleprice도 제거)
x = x.iloc[:, 1:-1]
y = house_train['SalePrice']

# 데이터 프레임 다 보이게
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

pd.reset_option('all')

# 결측값 확인
x.isna().sum()
# 결측치 있는 3개의 변수에 test의 mean값을 넣기
x['LotFrontage']=x['LotFrontage'].fillna(house_test['LotFrontage'].mean())
x['MasVnrArea']=x['MasVnrArea'].fillna(house_test['MasVnrArea'].mean())
x['GarageYrBlt']=x['GarageYrBlt'].fillna(house_test['GarageYrBlt'].mean())

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값 구해줌

# 회귀 직선의 기울기와 절편
model.coef_ # 기울기 = a값
model.intercept_ # 절편 = b값
slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x)
y_pred

# 테스트 데이터 예측
test_x = house_test.select_dtypes(include=[int, float])
test_x = test_x.iloc[:, 1:-1]

# 결측치 확인
test_x.isna().sum()
test_x['LotFrontage'].fillna(house_test["LotFrontage"].mean(), inplace=True)
test_x['MasVnrArea'].fillna(house_test["MasVnrArea"].mean(), inplace=True)
test_x['BsmtFinSF1'].fillna(house_test["BsmtFinSF1"].mean(), inplace=True)
test_x['BsmtFinSF2'].fillna(house_test["BsmtFinSF2"].mean(), inplace=True)
test_x['BsmtUnfSF'].fillna(house_test["BsmtUnfSF"].mean(), inplace=True)
test_x['TotalBsmtSF'].fillna(house_test["TotalBsmtSF"].mean(), inplace=True)
test_x['BsmtFullBath'].fillna(house_test["BsmtFullBath"].mean(), inplace=True)
test_x['BsmtHalfBath'].fillna(house_test["BsmtHalfBath"].mean(), inplace=True)
test_x['GarageYrBlt'].fillna(house_test["GarageYrBlt"].mean(), inplace=True)
test_x['GarageCars'].fillna(house_test["GarageCars"].mean(), inplace=True)
test_x['GarageArea'].fillna(house_test["GarageArea"].mean(), inplace=True)

# 테스트 데이터 집값 예측
pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission8.csv", index=False)

=======================================================================================
# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test=pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

# 회귀분석 적합(fit)하기
# house_train["GrLivArea"]   # 판다스 시리즈
# house_train[["GrLivArea"]] # 판다스 프레임
# 숫자형 변수만 선택하기
x = house_train.select_dtypes(include=[int, float])
# 필요없는 칼럼 제거하기
x = x.iloc[:,1:-1]
y = house_train["SalePrice"]

# 변수별로 결측값 채우기 (.mode()함수는 값이 여러개일수 있어서 시리즈로 반환)
x.isna().sum()
fill_values = {
    'LotFrontage': x['LotFrontage'].mean(),
    'MasVnrArea': x['MasVnrArea'].mean(),
    'GarageYrBlt': x['GarageYrBlt'].mean(),
}

x=x.fillna(value=fill_values)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값 구해줌

# 회귀 직선의 기울기와 절편
model.coef_ # 기울기 = a값
model.intercept_ # 절편 = b값

# 테스트 데이터 예측
test_x = house_test.select_dtypes(include=[int, float])
test_x = test_x.iloc[:, 1:] # 마지막에 saleprice 없으니까 -1 하면 안 됨

# 결측치 채우기
test_x=test_x.fillna(test_x.mean())

# 결측치 확인
test_x.isna().sum()
test_x.mean()

# 테스트 데이터 집값 예측
pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission9.csv", index=False)

=======================================================================================




































