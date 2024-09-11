## 제 10장 로지스틱 회귀분석

## 오즈의 개념
# 로지스틱 회귀분석: 확률이 오즈 -> 선형모형으로 모델링
# 확률의 오즈: 어떤 사건이 발생할 확률(A)과 그 사건이 발생하지 않을 확률 (Ac)

import pandas as pd
import numpy as np
admission_data=pd.read_csv('../data/admission.csv')

# GPA: 학점, GRE: 대학원 입학시험 (영어, 수학)

# 합격을 한 사건: Admit
# 입학 허가 (Admit) 확률 오즈 (Odds)는?
# P(Admit) = 합격 인원 / 전체 학생
p_hat=admission_data['admit'].mean() # 합격 확률
p_hat / (1 - p_hat) # 입학할 확률에 대한 odd
# 오즈가 1보다 작기 때문에, 입학할 확률보다 입학하지 않을 확률이 더 큼

# P(A): 0.5보다 큰 경우 -> 오즈비: 무한대에 가까워짐
# P(A): 0.5 -> 오즈비: 1
# P(A): 0.5보다 작은 경우 -> 오즈비: 0에 가까워짐
# 확률의 오즈비가 갖는 값의 범위: 0 ~ 무한대

# 범주형 변수 rank를 사용한 오즈 계산
# 각 범주별 입학에 대한 오즈를 계산
unique_ranks = admission_data['rank'].unique() # 1에서부터 4등급까지 존재

grouped_data = admission_data \
            .groupby('rank', as_index=False) \
            .agg(p_admit=('admit', 'mean'))
grouped_data['odds'] = grouped_data['p_admit'] / (1 - grouped_data['p_admit'])
grouped_data
# 1등급 학생들이 입학에 성공할 확률은 입학에 실패할 확률보다 18% 더 높음
# 나머지 등급의 학생들은 입학할 확률이 입학에 실패할 확률보다 더 낮다는 것을 확인

# 확률이 오즈비가 3이다 P(A)?
# P(A) / 1-P(A) -> p(A) = 3/4

# 오즈를 사용한 확률 연산
# p_hat = Odds / Odds + 1 (Odds = exp(B0+B1x))
1.178571 / (1.178571 + 1)


## 로그 오즈
# 로지스틱 회귀분석에서 로그 오즈를 모델링하여 선형 회귀를 수행
# 그 결과를 오즈비로 변환해 독립 변수가 사건 발생에 미치는 영향을 해석

# 확률 = 0 ~ 1
# 회귀 = −∞ ~ +∞

# 범위가 달라서 그대로 선형회귀 못 돌림 -> log 변환해서 범위를 회귀에 맞춤

# 회귀분석 결과로 나온 값은 로그오즈
# 로그오즈를 해석가능한 오즈로 바꾸기 위해 exp() 지수함수 사용

# exp(log odds)= odds ratio 오즈비

# 오즈값: 사건이 발생할 확률과 발생하지 않을 확률의 비율

# 오즈비: 독립 변수가 1단위 증가할 때, 사건이 발생할 오즈가 얼마나 변하는지
# x1이 한 단위 증가할 때마다 성공y의 odds 가 몇 배 증가하는지

# (오즈비가 1보다 크면 해당 변수의 증가가 사건의 발생할 확률을 높이는 효과,
# 1보다 작으면 사건이 발생할 확률을 낮추는 효과)

# Admission 데이터 산점도 그리기
# x:GRE, y:Admission
import seaborn as sns
sns.scatterplot(data=admission_data, x='gre', y='admit')
# 다른 산점도도 알아보기
sns.stripplot(data=admission_data,
                x='rank', y='admit', jitter=0.3, # 겹쳐보이는 애들 흩어지게
                alpha=0.3) # 투명도 조절
sns.scatterplot(data=grouped_data, x='rank', y='p_admit')
sns.regplot(data=grouped_data, x='rank', y='p_admit') # 회귀직선


## 로지스틱 회귀계수 예측 아이디어
odds_data = admission_data.groupby('rank').agg(p_admit=('admit', 'mean')).reset_index()
odds_data['odds'] = odds_data['p_admit'] / (1 - odds_data['p_admit'])
odds_data['log_odds'] = np.log(odds_data['odds'])
odds_data

sns.regplot(data=odds_data, x='rank', y='log_odds')

import statsmodels.api as sm
model = sm.formula.ols("log_odds ~ rank", data=odds_data).fit()

print(model.summary()) # intercept: 절편, rank의 coef: rank기울기
# y(종속변수: log odds)=-0.5675x(독립변수: rank) + 0.6327


## 로지스틱 회귀계수 해석 아이디어
# rank 1단위 증가하면 log odds 0.5675 감소
# 오즈비 Odds ratio: x가 한단위 증가하면 오즈는 어떻게 변하는지?
np.exp(0.5675) # 1.763851905037701
# rank가 한 단위 증가할때마다 odds가 이전 오즈의 약 절반 가량 감소

# 오즈를 이용한 확률 역산
# p(x_rank) = exp(B0 + B1x_rank) / 1+exp(B0 + B1x_rank)


## Python에서 로지스틱 회귀분석 하기
# 로지스틱 회귀분석을 통해 admit (입학 성공 여부) 예측
import statsmodels.api as sm

# admission_data['rank'] = admission_data['rank'].astype('category')
admission_data['gender'] = admission_data['gender'].astype('category')
# gender 변수는 범주형 변수라서 category type으로 변환해야 회귀분석 처리가 됨

model = sm.formula.logit("admit ~ gre + gpa + rank + gender", data=admission_data).fit()
# 종속 변수 admit을 독립 변수 gre, gpa, rank, gender로 예측
# fit(): 모델 학습 -> 각 변수에 대한 회귀계수 도출

print(model.summary())

# gre(0.0023) GRE 점수가 1점 증가 -> 합격 로그 오즈 0.0023 증가
# -> np.exp(0.0023)=2.17 -> 0.2% 증가

# gpa(0.7753) GPA 점수가 1점 증가 -> 합격 로그 오즈 0.7753 증가
# -> np.exp(0.7753)=2.17 -> 0.2% 증가

# 입학할 확률의 오즈가 np.exp(0.7753)=2.17배 된다
# -> 다른 변수들이 고정일 때, gpa가 한 단위 증가하면 오즈가 np.exp(0.7753)배=2.1배 증가
# 오즈가 두배가 됐다고 해서 확률이 두배가 되는것은 아님

np.exp(-0.0576) # 0.9440274829178357
# 성별이 남성인 학생은 여성 학생에 비해 합격 로그 오즈가 0.0578 낮다
# 여학생 그룹과 남학생 그룹의 합격에 대한 오즈비가 0.944로 1보다 작다
# -> 남성이 여성에 비해 입학 확률이 약간 더 낮다는 의미
# 하지만 p값이 0.80이라 이 변수의 계수는 통계적으로 유의하지 않다
# -> 성별이 합격 여부에 큰 영향을 미치지 않는다

# 1 → 2 → 2배, 100% 증가
# 100 → 123 → 1.23배, 23% 증가 
# exp(0.8032) = 2.23배, 123% 증가

# 예측 확률 계산할 땐 Gender 변수 넣어야 함(수식 그대로 사용해야 함)
# Gender 변수 자체는 통계적으로 유의미하지 않아서 해석이 안 되는 것

# 유의하지 않은 변수는 넣어서 써도 되긴 한데
# 빼고 싶다면 모델을 만들 때 변수를 빼고 다시 모델을 만들어야 함

# 유의확률: 각 변수의 계수가 0인지 아닌지
# H0: Bm=0 vs Ha: Bm!=0
# 0.800> 0.05-> H0 기각하지 못한다 (y값에 Bm의 영향이 x)


# -------------------------------------------------------------------------------
# 여학생
# GPA: 3.5
# GRE: 500
# Rank: 2
# 합격 확률 예측해보세요
log_odds = -3.4075 + 0.0023 * 500 + 0.7753 * 3.5 -0.5614*2
odds = np.exp(log_odds)
p_hat = odds / (odds + 1)
0.3392249517019396
# gender[T.M]: 남자와 여자의 intercept 차이, 여자가 0이고 남자가 1 (여학생이 base level)
# gender[T.M]이 남성을 나타내며, 기준이 되는 base level은 여학생임을 암시
# Odds는? 0.3392249517019396 / (1-0.3392249517019396)

# 이 상태에서 GPA가 1 증가하면 합격 확률 어떻게 변할까?
log_odds = -3.4075 + 0.0023 * 500 + 0.7753 * 4.5 -0.5614*2
odds = np.exp(log_odds)
p_hat = odds / (odds + 1)
p_hat # 합격확률: 0.5271108843656935
# odds는? 0.5271108843656935 / (1-0.5271108843656935)

# 여학생
# GPA: 3
# GRE: 450
# Rank: 2
# 합격 확률과 odds
log_odds = -3.4075 + 0.0023 * 450 + 0.7753 * 3 -0.5614*2
odds = np.exp(log_odds)
p_hat = odds / (odds + 1)
0.23696345421630416 # p_hat 합격확률
# 0.23696345421630416 / (1-0.23696345421630416)
0.31055321730746843 # odds

# 오즈가 오즈비만큼 변한다


## 각 계수 검정하기 with Wald test
# 귀무가설 Bi=0 대립가설 Bi!=0
# 유의수준 5% 하에서 gre계수가 0이라는 귀무가설을 기각할 수 있다

import scipy.stats as stats
z = 0.002256 / 0.001094 # z통계량: 2.0621572212065815
p_value = 2 * (1-stats.norm.cdf(z)) # p-value: 0.03919277001389343

# log(p(x) / (1-p(x)) (로그오즈) = B0 + B1x + .. BkXk (B0, B1은 로지스틱 회귀계수)
# p(x) / (1-p(x) = exp(B0 + B1x + .. BkXk) (로그오즈에 지수함수 exp())
# B1 의미 해석을 위해 odds_ratio 개념 생성
# X1의 odds_ratio = odds(x+1) / odds(x) = e^B1
# 오즈비: e^B → 로지스틱회귀계수같은

# ## LR test: 회귀모델이 유의한가?
# 모델의 유의성 체크
# 일반선형모형 - F검정
# 로지스틱회귀모형 - Likelihood Ratio 검정 (LR)
# H0: 모든 B 계수 0 (모델의 변수가 종속 변수에 아무런 영향을 미치지 x)
# Ha: 0이 아닌 B 계수 존재
# Λ = −2(ℓ(𝛽)̂ (0) − ℓ(𝛽)̂ ) ∼ 𝜒2 𝑘−𝑟 = 2×(log(L0)−log(L1))
# L0: 귀무가설 하에서의 로그 우도값 (모든 회귀계수가 0일 때의 모델 적합도)
# L1: 대립가설 하에서의 로그 우도값 (회귀모델에서 계수들이 0이 아닐 때의 모델 적합도)
# 위의 검정통계량은 카이제곱분포 자유도가 두 모델의 자유도 차를 따르게 되므로, p‑value를 구할 수 있음
# 자유도는 두 모델의 자유도의 차이 (이 값을 사용하여 p-value 구하면 됨)
# → LR 테스트는 귀무가설(모든B계수가0)을 기각하기 위한 검정,
# -> 계산된 결과에 따라 모델이 데이터를 잘 설명한다고 할 수 있으며,
# -> 모델이 통계적으로 유의미하다고 결론 내릴 수 있음