import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# y=2x+3의 그래프를 그려보세요!
a=2
b=3

x = np.linspace(-5, 5, 100)
y = a*x +b

plt.plot(x, y, color='blue')
plt.axvline(0, color='black')
plt.axhline(0, color='black')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()
plt.clf()

==========================================================================
# bedroom 갯수와 saleprice 관계
# a와 b 최적의 수 구해보기

# a = 70
# 방하나가 늘어나면 집값이 7천만원 늘어난다
# b = 10
# 방이 하나도 없어도 기본적으로 천만원

# a=70, b=10 가정
# 방 1개 집값: 8천만원
# 방 2개 집값: 1억 5천만원
# 방 3개 집값: 2억 2천
# 방 4개 집값: 2억 9천
# 방 5개 집값: 3억 6천

a=70
b=10
x = np.linspace(0, 5, 100)
y = a*x +b

house_train=pd.read_csv("./data/houseprice/train.csv")
my_df=house_train[['BedroomAbvGr','SalePrice']].head(10)
my_df['SalePrice']=my_df['SalePrice']/1000

plt.scatter(x=my_df['BedroomAbvGr'], y=my_df['SalePrice'], color='pink')
plt.plot(x, y, color='green')
plt.show()
plt.clf()

# Q. a=70, b=10 넣은 saleprice 만들어서 sub에 만들기

a=70
b=10

# 테스트 집 정보 가져오기 (saleprice 정보 x)
house_test =  pd.read_csv('./data/houseprice/test.csv')

(a * house_test['BedroomAbvGr'] + b) * 1000

# sub 데이터 불러오기
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

# SalePrice 바꿔치기
sub_df["SalePrice"] = (a * house_test["BedroomAbvGr"] + b) * 1000
sub_df

sub_df.to_csv("./data/houseprice/sample_submission4.csv", index=False)
==========================================================================
# 직선 성능 평가
a=12
b=170

# y: 실제 데이터 값(실제값), y_hat: 직선을 사용한 집값 (예측값)

# y^hat 어떻게 구할까?
y_hat=(a * house_train["BedroomAbvGr"] + b) * 1000
# y는 어디에 있는가?
y=house_train["SalePrice"]

np.abs(y - y_hat)  # 절대거리
np.sum(np.abs(y - y_hat)) # 절대값 합
np.sum((y - y_hat)**2) # 제곱합
# -> 이 값을 최소로 만들어주는 a, b를 구하고 그 직선을 '회귀직선'이라고 부름

==========================================================================
# 회귀분석 직선 구하기

!pip install scikit-learn

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 예시 데이터 (x와 y 벡터)
x = np.array([1, 3, 2, 1, 5]).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array([1, 2, 3, 4, 5])  # y 벡터 (레이블 벡터는 1차원 배열입니다)

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

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()
==========================================================================
# 회귀모델을 통한 집값 예측

# housedata의 a b 값을 구해보아라
# x를 방갯수

# 필요한 패키지 불러오기
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test =  pd.read_csv('./data/houseprice/test.csv')
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

# 회귀분석 적합(fit)하기
x = np.array(house_train['BedroomAbvGr']).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = house_train['SalePrice']/1000 # y 벡터 (레이블 벡터는 1차원 배열입니다)

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

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()

import numpy as np
from scipy.optimize import minimize

# 최소값을 찾을 다변수 함수 정의
def my_f(x):
    return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

# 초기 추정값
initial_guess = [0, 0]

# 최소값 찾기
result = minimize(my_f, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

==========================================================================
# 회귀직선 구하기

import numpy as np
from scipy.optimize import minimize # 최적화를 위한 함수

def line_perform(par): # 주어진 파라미터 par에 대한 회귀 직선의 성능 평가, 선형회귀모델 계수&절편
    y_hat=(par[0] * house_train["BedroomAbvGr"] + par[1]) * 1000 # par[0]: 기울기, par[1]: 절편
    y=house_train["SalePrice"]
    return np.sum(np.abs((y-y_hat))) # 예측된 값과 실제 값의 차이의 절대값을 모두 더한 값을 반환, 최소 절대 오차를 계산한 것

line_perform([36, 68]) # 성능 함수 테스트

# 초기 추정값
initial_guess = [0, 0]

# 최소값 찾기
result = minimize(line_perform, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)
==========================================================================
# 0802

# 1. y=x^2+3의 최소값이 나오는 입력값 구하기
def  my_f(x):
    return x**2+3

my_f(3)

import numpy as np
from scipy.optimize import minimize

# 초기 추정값
initial_guess = [0]

# 최소값 찾기
result = minimize(my_f, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# 2. z=x^2+y^2+3
def my_f2(x):
    return x[0]**2+x[1]**2+3
my_f2([1, 3])

# 초기 추정값
initial_guess=[-10, 3]

# 최소값 찾기
result = minimize(my_f2, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

# 3. f(x,y,z)=(x-1)^2+(y-2)^2+(z-4)^2+7
# 최솟값과 최솟값이 되는 자리

def my_f3(x):
    return (x[0]-1)**2 + (x[1]-2)**2 + (x[3]-4)**2 +7
my_f3([1, 2, 3])
initial_guess=[1, 2, 4]
result = minimize(my_f2, initial_guess)
result.fun
result.x
==========================================================================
# houseprice 이용해서
test_x = np.array(house_test['BedroomAbvGr']).reshape(-1, 1) 
# 예측 집 값을 어떻게 구한담?
pred_y=model.predict(test_x) # test 셋에 대한 예측 집 값
pred_y
sub_df
sub_df['SalePrice']=pred_y*1000


house_train=pd.read_csv("./data/houseprice/train.csv")
house_test = pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df = pd.DataFrame({'Id' : house_test['Id'],
                       'SalePrice' : house_test['Id']})
sub_df = pd.DataFrame({'Id' : house_test['Id'],
                       'SalePrice' : house_test['Id']})
sub_df.shape
house_test.shape
house_train.shape

# train 값으로 공부를 해서 test 값으로 시험을 친다
# 주어진 훈련 데이터셋(train data)으로 모델을 학습하고, 테스트 데이터셋(test data)의 값을 예측하는 방식
==========================================================================
# 어떤 변수가 점수가 더 높을까 ,,~?

# 우리조는 GarageArea 변수 이용 (null을 평균치 대체)

x = np.array(house_train['GarageArea']).reshape(-1, 1)  # x 벡터 (특성 벡터는 2차원 배열이어야 합니다)
y = np.array(house_train['SalePrice'])  # y 벡터 (레이블 벡터는 1차원 배열입니다)
# 선형 회귀 모델 생성
model = LinearRegression()
# 모델 학습
model.fit(x, y)               # 자동으로 기울기, 절편 값을 구해줌
# 회귀 직선의 기울기와 절편
model.coef_                   # 기울기 a
model.intercept_              # 절편 b
slope = model.coef_[0]
intercept = model.intercept_
# 예측값 계산
pred_y = model.predict(x)

house_test['GarageArea'].isna().sum()
house_test['GarageArea'].fillna(house_train['GarageArea'].mean(), inplace = True)
# inplace=True는 결측치 있을 경우에만 사용

test_x = np.array(house_test['GarageArea']).reshape(-1, 1)
y_pred_hat = model.predict(test_x)
sub_df['SalePrice'] = y_pred_hat

sub_df.to_csv("./data/houseprice/sample_submission4.csv", index=False)
==========================================================================
# GrLivArea 변수 사용
x = np.array(house_train['GrLivArea']).reshape(-1, 1)
y = np.array(house_train['SalePrice'])
model = LinearRegression()
model.fit(x, y) 
model.coef_
model.intercept_
slope = model.coef_[0]
intercept = model.intercept_
pred_y = model.predict(x)
house_test['GrLivArea'].isna().sum()
test_x = np.array(house_test['GrLivArea']).reshape(-1, 1)
y_pred_hat = model.predict(test_x)
sub_df['SalePrice'] = y_pred_hat
sub_df.to_csv("./data/houseprice/sample_submission5.csv", index=False)
==========================================================================
# 패키지 불러오기
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test = pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

# 이상치 탐색 (그럼 이상치 두개만 빼진 house_train data 생성)
house_train=house_train.query('GrLivArea<=4500')

# 회귀분석 적합(fit)하기
# x = np.array(house_train['GrLivArea']).reshape(-1, 1)
x=house_train[['GrLivArea']]
y = np.array(house_train['SalePrice'])

# 선형 회귀 모델 생성
model = LinearRegression()
# 모델 학습
model.fit(x, y)               # 자동으로 기울기, 절편 값을 구해줌
# 회귀 직선의 기울기와 절편
model.coef_                   # 기울기 a
model.intercept_              # 절편 b
slope = model.coef_[0]
intercept = model.intercept_
# 예측값 계산
pred_y = model.predict(x)

sub_df.to_csv("./data/houseprice/sample_submission6.csv", index=False)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, pred_y, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, 5000])
plt.ylim(0, 900000)
plt.legend()
plt.show()
plt.clf()
==========================================================================
# 원하는 변수 2개만 사용 (우선 숫자 변수만)
house_train['GarageArea'].isna().sum()
house_test['GarageArea'].isna().sum()

# 변수를 2개 써서 reshape(-1, 2) 사용하는데 [[]] 사용해서 데이터프레임으로 만들면 안 붙여도 됨
# x = np.array(house_train[['GrLivArea', 'GarageArea']]).reshape(-1, 2)
x = house_train[['GrLivArea', 'GarageArea']]
y = np.array(house_train['SalePrice'])

model = LinearRegression() # 선형회귀모델
model.fit(x, y) # 모델 학습
model.coef_# 기울기
model.intercept_ # 절편
slope = model.coef_[0]
intercept = model.intercept_
pred_y = model.predict(x) # 예측값
==========================================================================
# f(x, y)=ax+by+c
def my_houseprice(x,y):
     return model.coef_[0]*x+model.coef_[1]*y+model.intercept_

my_houseprice(300, 55) # 성능 함수 테스트

# GrLivArea, GarageArea를 x, y에 넣어서 데이터 프레임으로 만들어서
my_houseprice(house_test['GrLivArea'], house_test['GarageArea'])
test_x=house_test[['GrLivArea', 'GarageArea']]

# 결측치 확인
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()
test_x=test_x.fillna(house_test["GarageArea"].mean())


pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission7.csv", index=False)
==========================================================================
