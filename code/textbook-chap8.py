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

# plt.figure(figsize=(5, 4)) # 그래프 사이즈 조정

# 막대그래프
# mpg['drv'].unique() 데이터 유니크 값 찾아보기 
df_mpg = mpg.groupby('drv', as_index = False) \
    .agg(mean_hwy=('hwy', 'mean')) # as_index = False -> drv가 컬럼이 돼서 나온다
sns.barplot(data = df_mpg.sort_values('mean_hwy'),
            x = 'drv', y = 'mean_hwy',
            hue = 'drv')
plt.show()

df_mpg = mpg.groupby('drv', as_index = False) \
            .agg(n = ('drv', 'count'))
df_mpg

sns.barplot(data = df_mpg, x = 'drv', y = 'n')
plt.show()
sns.countplot(data = mpg, x = 'drv', hue = 'drv')

