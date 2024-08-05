# y-2x+3 그래프 그리기
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# y=2x+3을 기반으로 데이터를 생성하고 일부 랜덤 샘플을 선택하여 시각화

# x값의 범위 설정
x=np.linspace(0, 100, 400)

# y 값 계산
y=2*x+3

# x값의 범위 및 데이터 샘플링
# obs_x는 𝑦=2𝑥+3 직선과 함꼐 랜덤하게 선택된 20개의 집 크기, 임의로 선택된 집 크기
obs_x=np.random.choice(np.arange(100), 20)

# 오차항 생성 및 y 값 계산
# epsilon은 실제 데이터에서 관찰될 수 있는 불규칙한 변동을 모델링하기 위해 추가된 오차항
epsilon_i=norm.rvs(loc=0, scale=10, size=20) # 평균 0, 표준편차 10인 정규분포를 따르는 난수 20개 생성
obs_y=2*obs_x+3 +epsilon_i 
# obs_y는 obs_x에 대한 선형 함수 𝑦=2𝑥+3에 epsilon_i를 더해 계산된 값, 실제 관측된 y 값을 나타냄
# obs_y는 obs_x와 대응하는 y 값들이 오차항에 의해 변동된 모습을 보여줌

# 그래프 그리기
plt.plot(x, y, label='y=2x+3', color='black')
plt.scatter(obs_x, obs_y, color='blue', s=3)
plt.show()
plt.clf()

# 오차항을 넣는 이유는 실제 데이터가 모델에 의해 예측된 값과 항상 일치하지 않기 때문
# 다양한 요인들이 존재하기 때문에 실제 관측된 값(obs_y)은
# 단순한 선형 모델(y = 2x + 3)로 설명되지 않는 부분이 있습니다. 이 부분이 바로 오차항
==============================================================================================

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
obs_x=obs_x.reshape(-1, 1)
model.fit(obs_x, obs_y)

# 회귀 직선의 기울기와 절편
model.coef_ # 기울기 a hat
model.intercept_ # 절편 b hat

# 회귀 직선 그리기
x=np.linspace(0, 100, 400)
y=model.coef_[0]*x+model.intercept_
plt.xlim([0, 100])
plt.ylim(0, 300)
plt.plot(x, y, color='red') # 회귀직선
plt.show()
plt.clf()

import statsmodels.api as sm
obs_x=sm.add_constant(obs_x)
model =sm.OLS(obs_y, obs_x).fit()
print(model.summary())








