# 유의확률(p-value): 실제로는 집단 간 차이가 없는데 우연히 차이가 있는 데이터가 추출될 확률
# 유의확률이 작다면 집단 간 차이가 통계적으로 유의하다고 해석

====================================================================================
## t검정: 두 집단의 평균에 통계적으로 유의한 차이가 있는지 알아볼 때
import pandas as pd
mpg=pd.read_csv('./data/mpg.csv')

mpg.query('category in ["compact", "suv"]') \
    .groupby('category', as_index=False) \
    .agg(mean=('cty', 'mean'),
        count=('category', 'count'))

compact=mpg.query('category=="compact"')['cty']
suv=mpg.query('category=="suv"')['cty']

from scipy import stats
result=stats.ttest_ind(compact, suv, equal_var=True)
result.pvalue
result.statistic

mpg.columns

mpg.query('fl in ["r", "p"]') \
    .groupby('fl', as_index=False) \
    .agg(mean=('cty', 'mean'), n=('fl', 'count'))

regular=mpg.query('fl=="r"')['cty']
premium=mpg.query('fl=="p"')['cty']

result=stats.ttest_ind(regular, premium, equal_var=True)
result.statistic
result.pvalue
# -> p-value값이 0.28으로 유의수준 0.05보다 커서 H0을 기각하지 못한다
# 즉, 일반 휘발유와 고급 휘발유를 사용하는 자동차의 도시 연비 차이가 통계적으로 유의하지 않다
# 고급 휘발유 도시 연비 평균이 0.6 높지만 이건 우연히 발생했을 가능성(28%)이 크다고 해석
# 
# H0: 통계적으로 유의하지 않다 (= 차이가 없다 같다)
# Ha: 통계적으로 유의하다 (!= 차이가 있다)
====================================================================================
## 상관분석 - 두 연속 변수가 서로 관련이 있는지 검정

# 두 변수가 얼마나 관련되어 있는지, 관련성의 정도를 파악할 수 있음
# 0~1의 값을 가지며 1에 가까울 수록 관련성이 크다는 것을 의미
# 상관계수가 양수면 정비례, 음수면 반비례 관계를 의미

economics=pd.read_csv('./data/economics.csv')
economics[['unemploy', 'pce']].corr()
# 상관계수: 0.61-> 실업자수와 개인 소비 지출은 한 변수가 증가하면 다른 변수가 증가하는 정비례 관계

stats.pearsonr(economics['unemploy'], economics['pce'])
# 0.614517614193208(상관계수), 6.773527303290071e-61(유의확률)
# -> 유의확률이 유의수준 0.05보다 작기 때문에 실업자 수와 개인 소비 지출의 상관관계가 통계적으로 유의
====================================================================================
# 상관행렬: 모든 변수의 상관관계를 나타냄 (여러 변수의 관련성 한꺼번에 알아보기 위해)
mtcars=pd.read_csv('./data/mtcars.csv')
mtcars.head()
car_cor=mtcars.corr()

pd.set_option('display.max_columns', None)

# 히트맵 만들기
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi': '120',
                     'figure.figsize': [7.5, 5.5]})

import seaborn as sns
sns.heatmap(car_cor,
            annot=True,
            cmap='RdBu')
plt.show()
plt.clf()

# 대각 행렬 제거하기
import numpy as np
mask=np.zeros_like(car_cor)
mask[np.triu_indices_from(mask)]=1 # 오른쪽 위 대각 행렬을 1로 바꾸기

sns.heatmap(data=car_cor,
            annot=True,
            cmap='RdBu',
            mask=mask)
plt.show()
plt.clf()

mask_new=mask[1:, :-1]
cor_new=car_cor.iloc[1:, :-1]
sns.heatmap(data=cor_new,
            annot=True,
            cmap='RdBu',
            linewidths=5,
            vmax=1,
            vmin=-1,
            cbar_kws={'shrink': .5},
            mask=mask_new)
plt.show()
plt.clf()
