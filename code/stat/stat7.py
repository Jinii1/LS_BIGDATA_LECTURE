import numpy as np
import pandas as pd

## 2교시
# ADP p. 59 표 만들기
tab3=pd.read_csv('./data/tab3.csv')
tab3

tab1=pd.DataFrame({'id': np.arange(1, 13),
                   'score': tab3['score']})
                    
tab2=tab1.assign(gender=['female']*7 + ['male']*5)

## 1 표본 t 검정 (그룹 1개)
# 귀무가설 vs 대립가설
# H0: Mu=10 vs Ha:Mu!=10
# 유의수준 5%로 설정

from scipy.stats import ttest_1samp
t_statistic, p_value = ttest_1samp(tab1['score'], popmean=10, alternative='two-sided')
t_statistic
p_value

result=p_value = ttest_1samp(tab1['score'], popmean=10, alternative='two-sided')
t_value=result[0] # t 검정통계량: 표본 데이터가 모집단 평균과 얼마나 차이가 나는지를 측정
p_value=result[1] # 유의확률 (p-value): p-값이 작을수록 (일반적으로 0.05 이하) 귀무 가설을 기각

# t-값 (검정 통계량): t-검정에서 사용되는 통계량, 표본과 모집단 평균 간의 차이를 표준화한 값
# p-값 (유의 확률): 현재 표본 데이터가 관찰될 확률, 이 값이 작을수록 귀무 가설을 기각할 가능성이 커짐
# α (유의수준): 귀무가설 기각 기준, 보통 0.05(5%)

result.pvalue
result.statistic
result.df

# 귀무가설이 참일 때(Mu=10), 표본평균 11.53이 관찰될 확률이 6.48%이므로
# 이것은 우리가 생각하는 보기 힘들다고 판단하는 기준인 0.05(유의수준보다 크므로)
# 귀무가설이 거짓이라 판단하기 힘들다
# -> 유의확률 0.0648이 유의수준 0.05보다 크므로 귀무가설을 기각하지 못한다

# 95% 신뢰구간 구하기
ci=result.confidence_interval(confidence_level=0.95)
ci[0]
ci[1]
===========================================================================================
## 3교시

## 2표본 t 검정 (그룹2) - 분산 같고, 다를 때
# 분산 같은 경우: 독립 2표본 t검정
# 분산 다른 경우: Welch's t 검정
# 귀무가설 vs 대립가설
# H0: Mu_m = Mu_f, H1: Mu_m > Mu_f
# 유의수준 1%로 설정, 두 그룹의 분산은 같다고 가정

from scipy.stats import ttest_ind

f_tab2=tab2[tab2['gender']=='female']
m_tab2=tab2[tab2['gender']=='male']

result = ttest_ind(f_tab2['score'], m_tab2['score'],
            alternative='less', equal_var=True)
# f_tab(Mu_f)<m_tab2(Mu_m) 이라 alternative=less
# alternative='less'의 의미
# 대립가설이 첫 번째 입력그룹의 평균이 두번째 입력 그룹 평균보다 작다
# m_tab2['score'], f_tab2['score'] 라면, alternative='greater'

result.statistic
result.pvalue
# ci=result.confidence_interval(0.95)
# ci[0]
# ci[1]

## 대응표본 t 검정 (짝지을 수 있는 표본)
# 귀무가설 vs 대립가설
# H0: mu_before = mu_after vs Ha: mu_after > mu_before
# H0: mu_d = 0, Ha: mu_d > 0
# mu_d=mu_after-mu-before
# 유의수준 1%로 설정, 두 그룹의 분산은 같다고 가정

# mu_d에 대응하는 표본으로 반환
tab3_data = tab3.pivot_table(index='id',
                             columns='group',
                             values='score').reset_index()
tab3_data['score_diff'] = tab3_data['after'] - tab3_data['before']
test3_data = tab3_data[['score_diff']]
test3_data

result=p_value = ttest_1samp(test3_data['score_diff'], popmean=0, alternative='greater')
t_value=result[0] # t 검정통계량
p_value=result[1] # 유의확률 (p-value)

===========================================================================================
## 4교시

### long to wide: pivot_table()

tab3_data = tab3.pivot_table(
index='id',
columns='group',
values='score'
).reset_index()

### wide to long: melt()

long_form = tab3_data.melt(
id_vars='id',
value_vars=['before', 'after'],
var_name='group',
value_name='score')

# 연습1
df = pd.DataFrame({
    'id': [1, 2, 3],
    'A': [10, 20, 30],
    'B': [40, 50, 60]
})

df_long = df.melt(id_vars='id',
                    value_vars=['A', 'B'],
                    var_name='group',
                    value_name='score')

df_long.pivot_table(columns='group', values='score', aggfunc='mean')

# 연습2
import seaborn as sns
tips=sns.load_dataset('tips')

tips.pivot_table(columns='day', values='tip')
