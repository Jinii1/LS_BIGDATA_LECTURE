# pd, np, plt, sns .. 등등 Shift + spacebar

# 산점도
import pandas as pd
mpg = pd.read_csv('data/mpg.csv')
mpg
mpg.shape
                    
import seaborn as sns
! pip install seaborn
import matplotlib.pyplot as plt
sns.scatterplot(data = mpg, x = 'displ', y = 'hwy',
                            hue = 'drv') \
   .set(xlim = [3, 6], ylim = [10, 30])
plt.show()
plt.clf()

# p. 204
# Q1
import seaborn as sns
sns.scatterplot(data=mpg, x='cty', y='hwy')
plt.show()
plt.clf()

# Q2
midwest = pd.read_csv('data/mpg.csv')
midwest
midwest['poptotal']
sns.scatterplot(data = midwest, x = 'poptotal', y = 'popasian') \
   .set(xlim = [0, 500000], ylim = [0, 10000]) 
plt.show()

# plt.figure(figsize=(5, 4)) # 그래프 사이즈 조정

# 막대그래프
# mpg['drv'].unique() 데이터 유니크 값 찾아보기 
# 구동 방식별 고속도로 연비 평균
df_mpg = mpg.groupby('drv', as_index = False) \
            .agg(mean_hwy=('hwy', 'mean')) # as_index = False(변수를 인덱스로 만들지 않고 유지)
sns.barplot(data = df_mpg.sort_values('mean_hwy'),
            x = 'drv', y = 'mean_hwy',
            hue = 'drv')
plt.show()

# 크기순으로 정렬
df_mpg = df_mpg.sort_values('mean_hwy', ascending = False)
sns.barplot(data=df_mpg, x='drv', y='mean_hwy')
plt.show()
plt.clf()

# 집단별 빈도표 만들기
df_mpg = mpg.groupby('drv', as_index = False) \
            .agg(n = ('drv', 'count'))
sns.barplot(data = df_mpg, x = 'drv', y = 'n')
plt.show()
plt.clf()

sns.countplot(data = mpg, x = 'drv', hue = 'drv', order=['4', 'f', 'r'])
plt.show()
plt.clf()

# drv의 값을 빈도가 높은 순으로 출력
mpg['drv'].value_counts().index
sns.countplot(data=mpg, x='drv', order=mpg['drv'].value_counts().index)
plt.show()
plt.clf()

plt.show()
plt.clf()

# p. 211
# Q1
mpg
df = mpg.query('category=="suv"') \
   .groupby('manufacturer', as_index=False) \
   .agg(mean_cty=('cty', 'mean')) \
   .sort_values('mean_cty', ascending=False) \
   .head()

sns.barplot(data=df, x='manufacturer', y='mean_cty')
plt.show()
plt.clf()

# Q2
df2 = mpg.groupby('category', as_index=False) \
         .agg(n=('category', 'count')) \
         .sort_values('n', ascending = False)
df2

sns.barplot(data=df2, x='category', y='n')
plt.show()
plt.clf()

# 0729
# 선 그래프 - 시간에 따라 달라지는 데이터 표현
# 일정 시간 간격을 두고 나열된 데이터 -> 시계열 데이터 (ex. 환율, 주가지수 등 경제지표)
economics = pd.read_csv('data/economics.csv')
economics.head()
economics.info()

# 시간에 따라 실업자 수가 어떻게 변하는지 시계열 그래프로 표현
# lineplot() 선그래프
import seaborn as sns
sns.lineplot(data=economics, x='date', y='unemploy')
plt.show()
plt.clf()

# x축에 연도 표시하기
# 날짜 시간 타입 변수 만들기

# 날짜 시간 타입 변수 만들기
economics['date2'] = pd.to_datetime(economics['date'])
economics.info()
economics[['date', 'date2']] # 날짜 시간 타입으로 바꿔도 값은 같음
economics['date2'].dt.year # dt.month, dt.day -> 변수가 날짜 시간 타입으로 되어 있으면 df.dt로 추출
economics['date2'].dt.month
economics['date2'].dt.month_name()
economics_quarter = economics['date2'].dt.quarter

economics['quarter'] =  economics['date2'].dt.quarter
economics[['date2', 'quarter']]

# 각 날짜는 무슨 요일인가?
economics['date2'].dt.day_name()

# 각 날짜에 3일씩/ 한달씩 더하기
economics['date2'] + pd.DateOffset(days=3)
economics['date2'] + pd.DateOffset(days=30) # 하지만 한 달은 28~31일 ..
economics['date2'] + pd.DateOffset(months=1)

# 연도 변수 만들기
economics['year'] = economics['date2'].dt.year
economics.info()

# x축에 연도 표시하기
sns.lineplot(data=economics, x='year', y='unemploy')
plt.show()
sns.lineplot(data=economics, x='year', y='unemploy', errorbar=None) # 신뢰구간 (흐릿한 부분) 삭제
plt.clf()

# p. 217 혼자서 해보기
# Q1 시간에 따라 연도별 개인 저축률의 변화를 나타낸 시계열 그래프
sns.lineplot(data=economics, x='year', y= 'psavert', errorbar = None)
plt.show()

# Q2 2014년 월별 psavert의 변화를 나타낸 시계열 그래프
economics['month'] = economics['date2'].dt.month # 월 변수 추가
df_2014 = economics.query('year == 2014') # 2014년 추출
df_2014
sns.lineplot(data=df_2014, x='month', y='psavert', errorbar=None)
plt.show()

# 8.5 skip

# p. 161 참고
# 각각 해당년도의 표본평균과 표본편차 구해보기
my_df = economics.groupby('year', as_index=False) \
               .agg(mean_month  = ('unemploy', 'mean'),
                    std_month   = ('unemploy', 'std'),
                    count_month = ('unemploy', 'count'))
my_df

# 신뢰구간 만드는식: mean +/- 1.96*std/sqrt(12)
# errorbar를 설정하지 않고 p.217 그래프 그린 것
import numpy as np
my_df['left_ci'] = my_df['mean_month']-1.96*my_df['std_month']/np.sqrt(12) # 왼쪽 신뢰구간
my_df['right_ci'] = my_df['mean_month']+1.96*my_df['std_month']/np.sqrt(12) # 오른쪽 신뢰구간

import matplotlib.pyplot as plt
x = my_df['year']
y = my_df['mean_month']
plt.plot(x, y, color='black')
plt.show()
plt.clf()
