# pip install scipy
from scipy.stats import bernoulli


# 베르누이 분포 (p)
# 0과 1이 나오는 베르누이 확률변수의 결과값에 대응하는 확률분포
# P(X = k) = p ** k * (1 - p) ** 1 - k
# X는 베르누이 확률변수, 0 <= p <= 1
# p ** k는 1이 나오는 확률, (1 - p) ** 1 - k는 0이 나오는 확률

# 확률질량함수 (pmf)
# 확률변수가 갖는 값에 해당하는 확률을 저장하고 있는 함수
# bernoulli.pmf(k, p)
# P(X=1)
bernoulli.pmf(1, 0.3)
# P(0)
bernoulli.pmf(0, 0.3)

# 이항분포 X ~ P (X = k | n, p)

# 이항분포는 독립적인 베르누이 시행을 n번 반복하여 성공하는 횟수에 대한 분포
# 베르누이 시행의 성공 확률 p
# 𝑃 (𝑋 = 𝑘) = (𝑛 k) * 𝑝**𝑘 * (1 − 𝑝) ** 𝑛−𝑘
# n: 베루누이 확률변수 더한 갯수
# p: 1이 나올 확률
# (𝑛𝑘) = 𝑛! / 𝑘!(𝑛 − 𝑘)!

# binom.pmf()
from scipy.stats import binom

binom.pmf(0, n =2, p = 0.3)
binom.pmf(1, n =2, p = 0.3)
binom.pmf(2, n =2, p = 0.3)

# X ~ B (n, p)
# list comp.
result = [binom.pmf(x, n=30, p=0.3) for x in range (31)]
# numpy
binom.pmf(np.arange(31), n = 30, p = 0.3)

# ex2 이항분포 (nCr인데 54 C 26)
import math
math.factorial(54) / (math.factorial(26) * math.factorial(54-26))
math.comb(54, 26)

===============================================================
#  몰라도 됨

# 1*2*3*4
# np.cumprod(np.arange(1, 5))[-1]
fact_54 = np.cumprod(np.arange(1, 55))[-1]
# ln
# log (a*b) = log(a) + log(b)
# log(1 * 2 * 3 * 4) = log(1) + log(2) + log(3) +log(4)
math.log(24)
sum(np.log(np.arange(1, 5)))

math.log(math.factorial(54))
logf_54 = sum(np.log(np.arange(1, 55)))
logf_26 = sum(np.log(np.arange(1, 27)))
logf_28 = sum(np.log(np.arange(1, 29)))
# math.comb(54, 26)
np.exp(logf_54 - (logf_26 + logf_28))
===============================================================

math.comb(2, 0) * 0.3 ** 0 * (1-0.3) ** 2
math.comb(2, 1) * 0.3 ** 1 * (1-0.3) ** 1
math.comb(2, 2) * 0.3 ** 2 * (1-0.3) ** 0

# pmf 확률질량함수: probability mass function
# binom.pmf(k, n, p): n개의 원소 중에 k개의 원소룰 선택
binom.pmf(0, 2, 0.3)
binom.pmf(1, 2, 0.3)
binom.pmf(2, 2, 0.3)

# X ~ B (n=10, p=0.36) 일 때, P(X=4)?
binom.pmf(4, 10, 0.36)

# P(X <= 4)?
binom.pmf(np.arange(5), n =10, p =0.36).sum()
binom.cdf(4, 10, 0.36)

# P(2 < X <= 8)?
binom.pmf(np.arange(3,9), n=10, p=0.36).sum()
binom.cdf(8,10,0.36) - binom.cdf(2,10,0.36)

# X ~B (30, 0.2), P(X<4 or X>=25)
# method 1 (각각 구해서 더하기)
binom.pmf(np.arange(4), n=30, p=0.2).sum() + binom.pmf(np.arange(25, 31), n= 30, p =0.2).sum()

# method 2 (1 - 해당하지 않는 부분)
1 - sum(binom.pmf(np.arange(4, 25), n = 30, p = 0.2))

# rvs 함수 (random variates sample)
# 표본추출함수
# X1 ~ Bernoulli(p=0.3)
bernoulli.rvs(p = 0.3, size = 1)
# X2 ~ Bernoulli(p=0.3)
bernoulli.rvs(p = 0.3)
# X ~ B(n=2, p=0.3)
bernoulli.rvs(0.3) + bernoulli.rvs(0.3)
binom.rvs(n=2, p=0.3, size = 10)
binom.pmf(0, n = 2, p = 0.3)
binom.pmf(1, n = 2, p = 0.3)
binom.pmf(2, n = 2, p = 0.3)

# X ~ B (30, 0.26)의 표본 30개를 뽑아보세요 !
binom.rvs(n = 30, p = 0.26, size = 30)

# X ~ B (30, 0.26), E[X]?
binom.rvs(n = 30, p = 0.26, size = 30)

# 베르누이 확률변수의 기대값 E[Y] = p
# 이항분포 확률변수의 기대값 E[X] np

# X ~ B (30, 0.26) 시각화 해보세요
binom.rvs(n = 30, p = 0.26, size = 30)

import seaborn as sns
x = np.arange(31)
prob_x = binom.pmf(x, n = 30, p = 0.26)

sns.barplot(prob_x)
import matplotlib.pyplot as plt
plt.show()
plt.clf()


x = np.arange(31)
prox = binom.pmf(x, n = 30, p = 0.26)
import seaborn as sns
sns.barplot(prox)
plt.show()

# 교재 p. 207 참고해서 코드짜기
import pandas as pd
x = np.arange(31)
prob_x = binom.pmf(x, n = 30, p = 0.26)

df = pd.DataFrame({'x': x , 'prob': prob_x})
df

import seaborn as sns
sns.barplot(data = df, x = 'x', y = 'prob')
plt.show()

# CDF: cumulative dist. function (누적확률분포 함수)
# F_(X = x) = P(X <= x)

binom.cdf(4, n = 30, p = 0.26)

# P(4<X<=18) = ?
# = P(X<=18) - P(x<=4)
binom.cdf(18, n = 30, p = 0.26) - binom.cdf(4, n = 30, p = 0.26)

# P(13<X<20) = ?
# P(X<=19) - P(X<=13)
binom.cdf(19, n = 30, p = 0.26) - binom.cdf(13, n = 30, p = 0.26)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

x_1 = binom.rvs(n = 30, p = 0.26, size = 10) # 30번 시행, 성공확률 0.26, 10개의 랜덤 샘플
x_1
x = np.arange(31)
prob_x = binom.pmf(x, n = 30, p = 0.26)
sns.barplot(prob_x, color = 'coral')
plt.show()

# Add a point at (x_1, 0)
plt.scatter(x_1, np.repeat(0.002, 10), color = 'red', zorder = 100, s = 10) # s는 size
plt.show()
plt.clf()

# 기댓값 표현
plt.axvline(x = 7.8, color = 'green', linestyle = '--', linewidth = 2)
plt.show()
plt.clf()

# 총정리
# 확률질량함수 P(X=k) 확률변수가 어떤 값을 가지고 있는지
# 누적분포함수 P(X<=k)
# 랜덤샘플함수 random sample size
# 이항분포 X~B(n, p): 앞면(1)이 나올 확률이 p인 동전을 n번 던져서 나온 앞면의 수

# 퀸타일함수 ppf (cdf 반대 개념)
# P(X<?) = 0.5
binom.ppf(0.5, n = 30, p = 0.26) # 누적 확률이 0.5가 될 때 값을 반환
binom.cdf(8, n=30, p=0.26) # 8 이하일때의 누적 확률 계산
binom.cdf(7, n=30, p=0.26)

# P(X<?) = 0.7
binom.ppf(0.7, n=30, p=0.26)
binom.cdf(9, n=30, p=0.26)
binom.cdf(8, n=30, p=0.26)
    
1/np.sqrt(2*math.pi)
from scipy.stats import norm
norm.pdf(0, loc = 0, scale = 1)
# loc이 Mu, scale이 sigma
# 확률밀도함수 (PDF): 연속형 확률 변수 ex. 사람 키처럼 연속적으로 변하는 값들이 나올 확률
# 확률질량함수 (PMF): 이산형 확률 변수 ex. 주사위의 각 면이 나올 확률

# Mu=3, sigma=4, x=5라면?
norm.pdf(5, loc=3, scale=4)

# mu (평균) (loc): 분포의 중심

k = np.linspace(-5, 5, 100)
y = norm.pdf(k, loc=0, scale=1)

plt.plot(k, y, color='black')
plt.show()
plt.clf()

# sigma (표준편차) (scale): 분포의 퍼짐을 결정하는 모수 (모수: 특징을 결정하는 수)
k = np.linspace(-5, 5, 100)
y = norm.pdf(k, loc=0, scale=1)
y2 = norm.pdf(k, loc=0, scale=2)
y3 = norm.pdf(k, loc=0, scale=0.5)

plt.plot(k, y, color='black')
plt.plot(k, y2, color='purple')
plt.plot(k, y3, color='blue')

plt.show()
plt.clf()

# 평균이 0, 표준 편차가 1인 정규 분포에서 x = 0일 때 누적 확률, 결과: 0.5
norm.cdf(0, loc=0, scale=1) # 결과: 0.5
norm.cdf(100, loc=0, scale=1) # 결과: 1

# P(-2<X<0.54)=?
norm.cdf(0.54, loc=0, scale=1) - norm.cdf(-2, loc=0, scale=1)

# P(X<1 or X>3)=?
1 - norm.cdf(np.arange(4), loc = 0, scale = 1).sum

# 정규분포: Normal Distribution
# X~N(3, 5^2), P(3<X<5)=? 15.54%
norm.cdf(5, loc=3, scale=5) - norm.cdf(3, loc=3, scale=5)
norm.cdf(5, 3, 5) - norm.cdf(3, 3, 5)

# 위 확률변수에서 표본 1000개를 뽑아보자
norm.rvs(loc=3, scale=5 ,size = 1000) # rvs: 모수 정보를 넣고 size에 몇 개를 뽑을지
sum(((x>3) & (x<5)))/1000

# 표준정규분포: 평균 0, 표준편차 1, 표본 1000개 뽑아서 0보다 작은 비율 확인
x = norm.rvs(loc = 0,scale = 1,size = 1000)
np.mean(x<0)


x = norm.rvs(loc = 3,scale = 2,size = 1000)
x

sns.histplot(x, stat="density")

# Plot the normal distribution PDF
from scipy.stats import norm

# 정규 분포의 누적 분포 함수 값 계산
from scipy.stats import norm

xmin, xmax = (x.min(), x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc=3, scale=2)
plt.plot(x_values, pdf_values, color='red', linewidth=2)

plt.show()
plt.clf()
