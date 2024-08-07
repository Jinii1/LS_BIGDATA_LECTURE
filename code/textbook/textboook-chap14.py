# 유의확률(p-value): 실제로는 집단 간 차이가 없는데 우연히 차이가 있는 데이터가 추출될 확률
# 유의확률이 작다면 집단 간 차이가 통계적으로 유의하다고 해석

# t검정: 두 집단의 평균에 통계적으로 유의한 차이가 있는지 알아볼 때
import pandas as pd
mpg=pd.read_csv('./data/mpg.csv')

mpg.query('category in ["compact", "suv"]') \
    .groupby('category', as_index=False) \
    .agg(mean=('cty', 'mean'),
        count=('category', 'count'))

compact=mpg.query('category=="compact"')['cty']
suv=mpg.query('category=="suv"')['cty']

