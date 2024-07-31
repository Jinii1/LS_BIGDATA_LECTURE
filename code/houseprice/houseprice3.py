import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ExterCond: 주택 외부 재질의 현재 상태

# Ex: 매우 우수한 상태
# Gd: 좋은 상태
# TA: 평균적이거나 전형적인 상태
# Fa: 보통 이하의 상태
# Po: 나쁜 상태

# 가설: 일반적으로 외부 상태가 좋을수록 주택 가격이 높아지는 경향이 있을 것이라 예상,
# SalePrice가 Excellent, Good, Average, Fair, Poor 순으로 정렬될 것이다


train=pd.read_csv("./data/houseprice/train.csv")
train=train[['Id','ExterCond','SalePrice']]
train.info()

# 결측치 확인
train['ExterCond'].isna().sum()

train = train.groupby('ExterCond', as_index=False) \
            .agg(mean_sale=('SalePrice', 'mean'), count_sale=('SalePrice', 'count'))

train = train.sort_values('mean_sale', ascending=False)

sns.barplot(data=train, y='mean_sale', x='ExterCond', hue='ExterCond')
plt.show()
plt.clf()

train['mean_sale'].describe()

# 결과: 평균이라고 평가한 값이 두번째로 높은 것을 확인 -> 왜 그럴까 생각
# -> 각각의 표본 값을 세어 보니 poor과 excellent의 데이터 수가 너무 적은 것을 확인
# -> 유의미한 결론 도출 어려울수도 ..
===============================================================================================
0731에 배운 코드를 사용하여 변수 선택해서 팀별로 시각화
train =  pd.read_csv('/data/houseprice/train.csv')
#train = train[['Id', 'SaleType', 'ExterCond', 'GarageCars', 'LandContour', 'LandSlope', 'Neighborhood','SalePrice']]

# SaleType
SaleType_mean = train.groupby('SaleType', as_index = False) \
                     .agg(S_price_mean = ('SalePrice', 'mean'))
SaleType_mean = SaleType_mean.sort_values('S_price_mean', ascending = False)
sns.barplot(data = SaleType_mean, x = 'SaleType', y = 'S_price_mean', hue = 'SaleType')
plt.show()
plt.clf()

# ExterCond
ExterCond_mean = train.groupby('ExterCond', as_index=False) \
                      .agg(mean_sale=('SalePrice', 'mean'))

ExterCond_mean = ExterCond_mean.sort_values('mean_sale', ascending=False)

sns.barplot(data=ExterCond_mean, y='mean_sale', x='ExterCond', hue='ExterCond')
plt.show()
plt.clf()


# Ex > TA > Good> 각 개수가 몰려 있다.  




# GarageCars
GarageCars_mean = train.groupby('GarageCars', as_index = False) \
                             .agg(mean_price = ('SalePrice', 'mean')) \
                             .sort_values('mean_price', ascending = False)

sns.barplot(data = GarageCars_mean, x = 'GarageCars', y = 'mean_price', hue = 'GarageCars')
plt.show()
plt.clf()

gar1 = house_train.query('GarageCars == 1') \
           .sort_values('SalePrice', ascending = False).head(1000).mean()

gar2 = house_train.query('GarageCars == 2') \
           .sort_values('SalePrice', ascending = False).head(1000).mean()

gar3 = house_train.query('GarageCars == 3') \
           .sort_values('SalePrice', ascending = False).head(1000).mean()
           
gar4 = house_train.query('GarageCars == 4') \
           .sort_values('SalePrice', ascending = False).head(1000).mean()







LandContour_scatter1 = train[["LandContour", "SalePrice"]]
plt.scatter(data = LandContour_scatter1, x="LandContour", y= "SalePrice")
plt.show()
#
plt.clf()
LandContour_scatter2 = train[["LandSlope", "SalePrice"]]
plt.scatter(data = LandContour_scatter2, x="LandSlope", y= "SalePrice")
plt.show()


plt.clf()
LandContour_scatter3 = train[["SaleType", "SalePrice"]]
plt.scatter(data = LandContour_scatter3, x="SaleType", y= "SalePrice")
plt.show()



plt.clf()
LandContour_scatter5 = train[["Condition1", "SalePrice"]]
plt.scatter(data = LandContour_scatter5, x="Condition1", y= "SalePrice")
plt.show()


Neighborhood_mean = train.groupby('Neighborhood', as_index = False) \
                     .agg(N_price_mean = ('SalePrice', 'mean'))
####
plt.clf()
plt.grid()
sns.barplot(data = Neighborhood_mean, y = 'Neighborhood', x = 'S_price_mean', hue = 'Neighborhood')

LandContour_scatter4 = train[["Neighborhood", "SalePrice"]]
plt.scatter(data = LandContour_scatter4, y="Neighborhood", x= "SalePrice", s = 1, color = 'red')

plt.xlabel("price", fontsize=10)
plt.ylabel("n", fontsize=10)
plt.yticks(rotation=45,fontsize=8)
plt.show()


plt.clf()
sns.barplot(data = SaleType_mean, x = 'SaleType', y = 'S_price_mean', hue = 'SaleType')
plt.scatter(data = LandContour_scatter3, x="SaleType", y= "SalePrice", color = 'red')
plt.show()
