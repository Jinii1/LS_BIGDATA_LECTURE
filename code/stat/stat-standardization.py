from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

## 표본정규분포, 표본 1000개, 히스토그램 -> pdf 겹쳐서 그리기

# 표본 1000개의 표본정규분포의 히스토그램 그리기
# z는 정규분포를 따르는 (0, 1)
z = norm.rvs(loc = 0,scale = 1,size = 1000)
z

sns.histplot(z, stat="density")
plt.show()

# Plot the normal distribution PDF
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color='red', linewidth=2)

plt.show()
plt.clf()

# X~N(3, np.sqrt(2)^2) 으로 변경
# 2*np.sqrt(2)+3
x=z*np.sqrt(2)3
sns.histplot(z, stat='density', color='gray')
sns.histplot(x, stat='density', color='green')
plt.show()

zmin, zmax = (z.min(), x.max())
z_values = np.linspace(zmin, zmax, 500)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
pdf_values2 = norm.pdf(z_values, loc=3, scale=np.sqrt(2))
plt.plot(z_values, pdf_values, color='red', linewidth=2)
plt.plot(z_values, pdf_values2, color='blue', linewidth=2)

plt.show()
plt.clf()

============================================================================================
# 0731 오전 수업시간 표준화 필기 

## X ~ N(3, 7^2)

from scipy.stats import norm

x=norm.ppf(0.25, loc=3, scale=7)
z=norm.ppf(0.25, loc=0, scale=1)

x
3 + z * 7

norm.cdf(5, loc=3, scale=7)
norm.cdf(2/7, loc=0, scale=1)

norm.ppf(0.975, loc=0, scale=1)


z=norm.rvs(loc=0, scale=1, size=1000)
z

x=z*np.sqrt(2) + 3
sns.histplot(z, stat="density", color="grey")
sns.histplot(x, stat="density", color="green")
plt.show()

# Plot the normal distribution PDF
zmin, zmax = (z.min(), x.max())
z_values = np.linspace(zmin, zmax, 500)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
pdf_values2 = norm.pdf(z_values, loc=3, scale=np.sqrt(2))
plt.plot(z_values, pdf_values, color='red', linewidth=2)
plt.plot(z_values, pdf_values2, color='blue', linewidth=2)

plt.show()
plt.clf()
============================================================================================

# X 표본 1000개 뽑음 -> (표준화 사용) x를 z로 변경
# -> z의 히스토그램 그리기 -> 표준정규분포 pdf 겹쳐보기 (딱 겹쳐지면 표준화임을 알 수 있기 때문)

## Q1. X~N(5, 3^2)
x = norm.rvs(loc = 5,scale = 3,size = 1000)

# 표준화 만들기
z=(x-5)/3
sns.histplot(z, stat='density', color='gray')
plt.show()

# Plot the normal distribution PDF
zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color='red', linewidth=2)
plt.show()
plt.clf()
# -> 표준화 잘 작동하는걸 알 수 있음

## Q2. 표본표준편차로 나눠도 표준정규분포가 될까?
# 의도: 표본을 10개만 뽑은

# 1. x 표본을 10개를 뽑아서 표본분산값 계산
x= norm.rvs(loc = 5,scale = 3,size = 10)
s=np.std(x, ddof=1)
s

# 2. X 표본을 1000개 뽑음
x= norm.rvs(loc = 5,scale = 3,size = 1000)

# 3. 1번에서 계산한 s^2으로 sigma^2 대체한 표준화를 진행 (Z=X-Mu/s)
z=(x-5)/s

# 4. z의 히스토그램 그리기 -> 표준정규분포 pdf 그리기
sns.histplot(z, stat='density', color='gray')
plt.show()

zmin, zmax = (z.min(), z.max())
z_values = np.linspace(zmin, zmax, 100)
pdf_values = norm.pdf(z_values, loc=0, scale=1)
plt.plot(z_values, pdf_values, color='red', linewidth=2)
plt.show()
plt.clf()

# 결과: 이렇게 되면 표준정규분포가 아니다
============================================================================================
## t 분포
# X ~ t(n) 베르누이처럼 모수가 하나, 연속형 확률변수, 정규분포랑 비슷하게 생김
# -> 종모양, 대칭분포, 중심 0
# 모수 n: 자유도라고 부름 -> 퍼짐을 나타내는 모수 (자유도가 분산에 영향을 미침)
# n이 작으면 분산 커짐
# n이 무한대로 가면 표준정규분포가 된다

from scipy.stats import t

# t.pdf
# t.ppf
# t.cdf
# t.rvs
# -> 가능

## 자유도가 4인 t분포의 pdf를 그려보세요
# 자유도가 4인 t-분포의 확률 밀도 함수 (PDF)를 시각화

t_values = np.linspace(-4, 4, 100) # -4에서 4까지의 범위를 갖는 100개의 균등한 값으로 구성된 배열
pdf_values = t.pdf(t_values, df=4)
# 자유도가 4인 t-분포의 확률 밀도 함수를 계산 -> 주어진 x 값들에 대해 t-분포의 PDF 값을 반환
plt.plot(t_values, pdf_values, color='red', linewidth=2)

plt.show()
plt.clf()

# 표준정규분포 겹치기 (정규분포와 실제로 유사한지 알아보기 위해)
pdf_values = norm.pdf(t_values, loc=0, scale=1)
plt.plot(t_values, pdf_values, color='black', linewidth=2)

plt.show()
plt.clf()

# -> t분포가 꼬리가 길다 = 극단치 값이 더 많이 생긴다
# -> 자유도가 커질수록 표준정규분포랑 거의 비슷해짐

# 그래서 t분포는 왜 씀 ..?
# chatgpt) 표본 크기가 작거나 모분산이 알려지지 않은 경우

# X ~?(mu, sigma^2)
# X bar ~ N(mu, sigma^2/n) 표본평균의 분포
# X bar ~= t(x_bar, s^2/n) sigma를 s로 바꿔치기해서 N이 t로 바뀜, 자유도가 n-1인 t분포
x=norm.rvs(loc=15, scale=3, size=16, random_state=42)
x
x_bar=x.mean()
n=len(x)

# 모분산을 모를 때: 모평균에 대한 95% 신뢰구간을 구해보자
x_bar + t.ppf(0.975, df=n-1) * np.std(x, ddof=1) / np.sqrt(n)
x_bar - t.ppf(0.975, df=n-1) * np.std(x, ddof=1) / np.sqrt(n)

# 모분산(3^2)을 알 때: 모평균에 대한 95% 신뢰구간을 구해보자
x_bar + norm.ppf(0.975, loc=0, scale=1) * 3 / np.sqrt(n)
x_bar - norm.ppf(0.975, loc=0, scale=1) * 3 / np.sqrt(n)
