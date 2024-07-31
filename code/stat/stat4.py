import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np


# 특정 구간의 값이 동일한 가능성을 갖는 균일확률변수의 경우
# X ~ U (a, b) 균일분포, a 시작점 b 끝점 -> 확률밀도함수
# X ~ U (2, 6) 나올 수 있는 값이 2~6, 모든 수가 균일하게
# uniform에선 loc은 구간시작점, scale은 구간 길이

# X ~ 균일분포 U(a, b)
# loc:a, scale:b-a
uniform.rvs(loc=2, scale=4, size=1)
# 2부터 6까지 랜덤변수 1개 추출

# uniform.pdf(x, loc=0, scale=1)
# uniform.cdf(x, loc=0, scale=1)
# uniform.ppf(q, loc=0, scale=1)
# uniform.rvs(loc=0, scale=1, size=None, random_state=None)


k = np.linspace(0, 8, 100)
y = uniform.pdf(k, loc=2, scale=4)
plt.plot(k, y, color='pink') # x축, y축
plt.show()
plt.clf()
# 2에서부터 6까지의 숫자, 연속형 확률밀도함수니까 평행 라인으로 그래프가 그려짐
# 확률을 의미하니까 넓이가 1 

# P(X<3.25)일 때
uniform.cdf(3.25, loc=2, scale=4)

# P(5<X<8.39)=?
uniform.cdf(8.39, loc=2, scale=4) - uniform.cdf(5, loc=2, scale=4)

# 상위 7% 값은?
uniform.ppf(0.93, loc=2, scale=4)

uniform.rvs(loc=2, scale=4, size=20, random_state=42).mean()
# random_state는 숫자를 다 같게(매번 다르게 x)

# 표본 20개 뽑고 표본 평균 계산
x = uniform.rvs(loc=2, scale=4, size=20*1000, random_state=42)
x.shape
x = x.reshape(-1, 20)
x.shape
blue_x=x.mean(axis=1)
blue_x

sns.histplot(blue_x, stat='density')
plt.show()
plt.clf()

# X bar ~ N (Mu, sigma^2/n) (파란 벽돌)
# X bar ~ N (4, 1.33333333/20)
uniform.var(loc=2, scale=4) # 분산
uniform.expect(loc=2, scale=4) # 기댓값

# Plot the normal distribution PDF
plt.clf()
xmin, xmax = (blue_x.min(), blue_x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc=4, scale=np.sqrt(1.333333333333/20))
plt.plot(x_values, pdf_values, color='red', linewidth=2)
plt.show()

# 신뢰구간
# X bar ~ N (Mu, sigma^2/n) (파란 벽돌)
# X bar ~ N (4, 1.33333333/20)

# 신뢰구간 (a, b) 95%
norm.ppf(0.025, loc=4, scale=np.sqrt(1.333333333333/20)) # ppf는 왼쪽 기준
norm.ppf(0.975, loc=4, scale=np.sqrt(1.333333333333/20))

# 신뢰구간 (a, b) 99%
norm.ppf(0.005, loc=4, scale=np.sqrt(1.333333333333/20))
norm.ppf(0.995, loc=4, scale=np.sqrt(1.333333333333/20))

# plot the normal distrubution PDF
x_values = np.linspace(3, 5, 100)
pdf_values = norm.pdf(x_values, loc=4, scale=np.sqrt(1.333333333333/20))
plt.plot(x_values, pdf_values, color='red', linewidth=2)
plt.show()

# 표본평균(파란벽돌) 점찍기
blue_x = uniform.rvs(loc=2, scale=4, size=20).mean()
# norm.ppf(0.975, loc=0, scale=1) == 1.96
a= blue_x + 1.96 * np.sqrt(1.3333333/20)
b = blue_x - 1.96 * np.sqrt(1.3333333/20)
plt.scatter(blue_x, 0.002, color = 'blue', zorder = 10, s = 10)
plt.show()

plt.axvline(x=a, color='blue', linestyle='--', linewidth=2)
plt.axvline(x=b, color='blue', linestyle='--', linewidth=2)
norm.ppf(0.995, loc=0, scale=1)

# 기대값 표현
plt.axvline(x = 4, color = 'green', linestyle = '-', linewidth = 2)
plt.show()

plt.clf()

norm.ppf(0.025, loc=4, scale= np.sqrt(1.3333333/20))
norm.ppf(0.975, loc=4, scale= np.sqrt(1.3333333/20))

sns.histplot(blue_x, stat='density')
plt.show()






# Hw4
# 숙제 표본 분산 n-1 vs. n
# 표본 분산 계산 시 왜 n-1로 나누는지 알아보도록 하겠습니다.
# 균일분포 (3, 7)에서 20개의 표본을 뽑아서 분산을 2가지 방법으로 추정해보세요.
# n-1로 나눈 것을 s_2, n으로 나눈 것을 k_2로 정의하고, s_2의 분포와 k_2의 분포를 그려주세요! (10000개 사용)
# 각 분포 그래프에 모분산의 위치에 녹색 막대를 그려주세요.
# 결과를 살펴보고, 왜 n-1로 나눈 것을 분산을 추정하는 지표로 사용하는 것이 타당한지 써주세요!

x = uniform.rvs(loc=3, scale=4, size=20*10000)
x
x.shape
x=x.reshape(-1, 20)
x
x.shape

s_2 = np.var(x, ddof=1, axis=1)
s_2.shape

sns.histplot(s_2, stat='density')
plt.show()

uniform.var(loc=2, scale=4) # 모분산
plt.axvline(x = uniform.var(loc=2, scale=4), color = 'green', linestyle = '-', linewidth = 2)
plt.show()
plt.clf()

k_2 = np.var(x, ddof=0, axis=1)
k_2.shape

sns.histplot(k_2, stat='density')
plt.show()

uniform.var(loc=2, scale=4) # 모분산
plt.axvline(x = uniform.var(loc=2, scale=4), color = 'green', linestyle = '-', linewidth = 2)
plt.show()
