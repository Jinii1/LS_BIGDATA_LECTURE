import pandas as pd
import numpy as np
from scipy.stats import uniform
import seaborn as sns

# 이번달 자리 바꾸기
old_seat = np.arange(1, 29)
np.random.seed(20240729)
# 1~28 숫자 중에서 중복 없이 28개 숫자를 뽑는 방법
new_seat = np.random.choice(old_seat, 28, replace = False)
result = pd.DataFrame({'old_seat': old_seat,
              'new_seat': new_seat})
result.to_csv(path_or_buf = 'result.csv', sep=',', index=False)

# y = 2x 그래프 그리기
x = np.linspace(0, 8, 2)
y = 2 * x

# plt.scatter(x, y, s=3)
plt.plot(x, y, color='black')
plt.show()
plt.clf()

# y = x^2를 점 3개 사용해서 그리기
x = np.linspace(-8, 8, 100) # -8을 하는 이유: 그래프가 0을 중심으로 볼록
y = x ** 2
# plt.scatter(x, y, s = 3) 산점도로 표현

# x, y축 범위 설정
plt.xlim(-10, 10)
plt.ylim(0, 40)

# 비율 맞추기: 그래프 왜곡이 심함 (x축과 y축 비율이 안맞음)
plt.axis('equal')
plt.gca().set_aspect('equal', adjustable = 'box')
plt.show()
plt.clf()

==============================================================================================================
import numpy as np
import pandas as pd
from scipy.stats import norm

# ADP p.57 연습문제
# 신뢰구간문제 2번
# 작년 남학생 3학년 전체 분포의 표준편차는 6kg 이었다고 합니다.
# 이 정보를 이번 년도 남학생 분포의 표준편차로 대체하여 모평균에 대한 90% 신뢰구간을 구하세요.
x = np.array([79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8])
x.mean() # x bar: 68.89375
len(x) # n: 16

# x bar: 68.89375 , n: 16, sigma: 6, alpha: 0.1, 1 - alpha: 0.9 (신뢰수준), z(alpha/2) = Z(0.05)
# Z 0.005 의미: 정규분포 (Mu=0, sigma^2=1)에서(=표준정규분포에서), 상위 5%에 해당하는 x값
# -> norm.ppf(0.95, loc=0, scale=1)

z_005 = norm.ppf(0.95, loc=0, scale = 1)
z_005

# 신뢰구간
# 𝐶.𝐼.  = 𝑋 (X bar) ± 𝑧𝛼/2 * 𝜎/√𝑛
x.mean() + z_005 * 6 / np.sqrt(16) # 오른쪽
x.mean() - z_005 * 6 / np.sqrt(16) # 왼쪽
# 결과: (66.4264695595728, 71.3610304404272)

# 데이터로부터 E[X^2] 구하기
x = norm.rvs(loc=3, scale=5, size=100000)
np.mean(x**2) # sum(x**2) / (len(x) -1)

# Var(X) = E[X^2] - E[X]^2
# 25 = ? - 3^2
# E[X^2] = 25 + 3^2 = 34

# E[(X-X^2)/(2X)] 구해보기
x = norm.rvs(loc=3, scale=5, size=100000)
(x - x**2)/(2*x)
np.mean((x - x**2)/(2*x))

# 몬테카를로 적분
# 확률변수 기대값을 구할 때, 표본을 많이 뽑은 후,
# 원하는 형태로 변형하고 평균을 계산해서 기대값을 구하는 방법

# X~N(3, 5^2), Var(X) = sigma^2 = 25
# 표본 10만개 추출해서 S^2을 구해보세요
# 이것이 표본분산

np.random.seed(20240729)
x = norm.rvs(loc=3, scale =5, size=100000)
x_bar = x.mean()
s_2 = sum((x - x_bar) ** 2) / (100000-1)
s_2
# np.mean((x-x_bar)**2) (표본 분산)
np.var(x, ddof=1) # n-1으로 나눈 값 (표본 분산)
# np.var(x) 사용하면 안됨 주의! # n으로 나눈 값

# n-1 vs. n
x=norm.rvs(loc=3, scale=5, size=20)
np.var(x)
np.var(x, ddof=1)
