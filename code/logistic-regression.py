# 로지스틱 회귀분석은 확률이 오즈를 선형모형으로 모델링하는 개념
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
# p_hat = Odds / Odds + 1
1.178571 / (1.178571 + 1)

# 로그 오즈
# log(p / 1-p) 확률의 오즈의 로그
# 값의 범위: - 무한대 ~ 무한대

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

odds_data = admission_data.groupby('rank').agg(p_admit=('admit', 'mean')).reset_index()
odds_data['odds'] = odds_data['p_admit'] / (1 - odds_data['p_admit'])
odds_data['log_odds'] = np.log(odds_data['odds'])
odds_data

sns.regplot(data=odds_data, x='rank', y='log_odds')

import statsmodels.api as sm
model = sm.formula.ols("log_odds ~ rank", data=odds_data).fit()
print(model.summary()) # intercept: 절편, rank의 coef: rank기울기
# y(종속변수: log odds)=-0.5675x(독립변수: rank) + 0.6327
# =========================================================================
np.exp(0.5675)
# 확률이 주어지면 odds 계산 (0~무한대)
# 확률 증가 odds 증가 (같은 방향으로 움직이는 성향 띔)
# rank 증가 odds 감소 -> 확률이 반절로 줄어드는건 아니지만 입학 확률이 반절로 줄어듦

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

# gpa 회귀계수 0.8 -> gpa 1점 증가할 때 입학 로그 오즈가 0.8 증가
# -> 입학할 오즈가 exp(0.8)=2.23배 증가 = 0.2% 증가

np.exp(0.0023)

입학할 확률의 오즈가 
np.exp(0.7753) # 2.17

# 입학할 확률의 오즈가 np.exp(0.7753)배 된다
# -> 다른 변수들이 고정일 때, gpa가 한 단위 증가하면 오즈가 np.exp(0.7753)배=2.1배 증가
# 오즈가 두배가 됐다고 해서 확률이 두배가 되는것은 아님

# gender[T.M]의 회귀계수가 -0.0578이라면, 남성이 여성에 비해 입학 확률이 약간 더 낮다는 의미지만, p-value가 높다면 이 값은 유의미하지 않다는 뜻

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

# 로그오즈 → 오즈비로 변환하기 위해 exp() 함수를 사용합니다.
# 오즈비는 독립변수가 1 단위 증가할 때, 사건(예: 입학 성공)이 발생할 확률이 어떻게 변화하는지 직관적으로 보여줍니다.
# ----------------------------------------------------
