x = np.arange(2,12)
x_mean = sum(x)/ len(x)
x_var = sum((x-x_mean)**2)/len(x)

import numpy as np
x=np.arange(2, 13)
prob = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]) / 36
Ex =sum(x * prob)
V=sum(((x-Ex)**2)*prob)

2*Ex+3
np.sqrt(4*V)
# ==============================================
from scipy.stats import binom
binom.pmf(np.array([0, 1, 2, 3]), 3, 0.7)


# Y~B(20, 0.45)
# P(6<Y<=14)=?
sum(binom.pmf(np.arange(7,15), 20, 0.45))
binom.cdf(14, 20, 0.45) - binom.cdf(6, 20, 0.45)


# X~N(30,4^2)
# P(X>24)=?
from scipy.stats import norm
1-norm.cdf(24, loc=30, scale=4)

# 표본은 8개를 뽑아서 표본평균 X_bar 계산
# P(28<X_bar<29.7)=?

# 방법1
# X_bar _ N(30, 4^2/8)
a=norm.cdf(29.7, loc=30, scale=np.sqrt(4**2/8))
b=norm.cdf(28, loc=30, scale=np.sqrt(4**2/8))
a-b

# 방법2
# 표준화 사용 방법
mean = 30
s_var = 4/np.sqrt(8) # 자유도 
right_x = (29.7 - mean) / s_var # 표준화
left_x = (28 - mean) / s_var

a = norm.cdf(right_x, 0, 1) # 표준정규분포 기준
b = norm.cdf(left_x, 0, 1)
a-b

# ==============================================
# 자유도가 7인 카이제곱분포 확률밀도 함수 그리기
from scipy.stats import chi2


k=np.linspace(0, 8, 100)
y=uniform.pdf(k, loc=2, scale=4)
plt.plot(k, y, color='black')
# ==============================================
mat_a=np.array([14, 4, 0, 10]).reshape(2,2)
mat_a

# 귀무가설: 두 변수 독립
# 대립가설: 두 변수가 독립 X
# (여기서 두 변수란 운동선수 유무와 흡연 유무)

from scipy.stats import chi2_contingency

chi2, p, df, expected = chi2_contingency(mat_a)
chi2.round(3) # 검정 통계량
p.round(4) # p-value

# 유의수준 0.05이라면,
# p값이 0.05보다 작으므로, 귀무가설 기각
# 즉, 두 변수는 독립 아니다