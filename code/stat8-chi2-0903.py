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

chi2, p, df, expected = chi2_contingency(mat_a, correction=False)
chi2.round(3) # 검정 통계량
p.round(4) # p-value

np.sum((mat_a - expected)**2 / expected)

# 유의수준 0.05이라면,
# p값이 0.05보다 작으므로, 귀무가설 기각
# 즉, 두 변수는 독립 아니다
# X ~ chi2(1) 일 때, P(X > 12.6)=?
from scipy.stats import chi2

1-chi2.cdf(15.556, df=1)
p

# 귀무가설: 두 도시의 음료 선호도가 동일하다
# 대립가설: 두 도시의 음료 선호도가 동일하지 않다
mat_b = np.array([[50, 30, 20],
          [45, 35, 20]])

mat_b

chi2, p, df, expected = chi2_contingency(mat_b, correction=False)
chi2.round(3) # 검정 통계량
p.round(4) # p-value
expected
# ==============================================
## 5.5 적합도 검정
# 요일별 출생아 수
# H0: 요일별 신생아 출생 비율이 같다 (p1, p2 ... = 1/7)
# H1: 요일별 신생아 출생 비율이 같지 않다

from scipy.stats import chisquare
import numpy as np
from scipy.stats import chi2_contingency

observed=np.array([13, 23, 24, 20, 27, 18, 15])
expected=np.repeat(20, 7)
statistic, p_value = chisquare(observed, f_exp=expected) # 검정통계량
statistic.round(3) # 7.6
p_value.round(3) # 0.269

# P-value 0.269가 유의수준 5%보다 크므로 귀무가설을 기각하지 못한다
# 즉 각 요일별 신생아 출생 비율이 모두 같다

expected

# 각 셀의 기대빈도가 5보다 모두 크므로, 카이제곱검정의 결과 신뢰 가능
# ==============================================
## 5.7 연습문제
# 3번. 지역별 대선 후보의 지지율
# 순서: 귀무가설, 대립가설 설정하고 데이터 입력하고 검정통계량 계산
# 데이터 입력할 때 그룹이 행, 성향이 열 형식으로

# H0: 후보 A의 지지에 있어서 선거구 간의 차이는 없다 (선거구별 지지율 동일하다)
# H1: 후보 A의 지지에 있어서 선거구 간의 차이는 있다

mat_b = np.array([[176, 124],
                  [193, 107],
                  [159, 141]])
mat_b

chi2, p, df, expected = chi2_contingency(mat_b, correction=False)
chi2.round(3) # 검정 통계량
p.round(4) # p-value
# 유의수준 0.05보다 p값이 크므로, 귀무가설을 기각할 수 없다

expected