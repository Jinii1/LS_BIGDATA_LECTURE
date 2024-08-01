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

# a = 70 가정
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
y_hat=(a*house_df['BedroomAbvGr']+b)*1000
# y는 어디에 있는가?
y=house_df['SalePrice']

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
hous_train=pd.read_csv("./data/houseprice/train.csv")
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

def line_perform(par): # 주어진 파라미터 par에 대한 회귀 직선의 성능 평가
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
