import pandas as pd
exam = pd.read_csv('C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project1/exam.csv')
exam.describe()

mpg.head() # 데이터 앞부분 (형태) 확인
mpg.tail() # 데이터 뒷부분 확인
mpg.shape # 데이터 프래임의 행, 열 개수 출력
mpg.info() # 데이터 속성 파악
mpg.describe(include = 'all') #요약 통계량 -> 변수 특징 파악

df_raw = pd.DataFrame({'var1' : [1, 2, 1],
'var2' : [2,3, 2]})
df_raw
df_new = df_raw.copy()
df_new = df_new.rename(columns = {'var2' : 'v2'})

mpg
mpg_new = mpg.copy()
mpg_new = mpg_new.rename(columns = {'cty' : 'city', 'hwy' : 'highway'})
mpg_new.head()

import pandas as pd
df = pd.DataFrame({'var1': [4, 3, 8],
'var2': [2, 6, 1]})
df
df['var_sum'] = df['var1'] + df['var2']
df
df['var_mean'] = df['var_sum'] / 2
df

import pandas as pd
mpg = pd.read_csv('C:/Users/USER/Downloads/mpg.csv')
mpg['total'] = (mpg['cty'] + mpg['hwy']) / 2
mpg.head()
sum(mpg['total']) / len(mpg)
mpg['total'].mean()

import pandas as pd
exam = pd.read_csv('C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project1/data/exam.csv')
exam.head()
exam.head(10)
exam.tail()
exam.shape
exam.info()
exam.describe()

mpg = pd.read_csv('C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project1/data/mpg.csv')
mpg.head()
mpg.tail()
mpg.shape
mpg.info()
mpg.describe()

df = pd.DataFrame({'x': [1, 2, 3]})
df.head()
df.info()

df_raw = pd.DataFrame({'var1': [1, 2, 1],'var2': [2, 3, 2]})
df_raw
df_new = df_raw.copy()
df_new
df_new = df_new.rename (columns = {'var1' : 'v1'})

df = pd.DataFrame({'var1': [4, 3, 8],'var2': [2, 6, 1]})
df

df['var_sum'] = df['var1'] +df['var2']
df

df['var_mean'] = df['var_sum'] / 2
df

mpg
mpg['total'] = (mpg['cty'] + mpg['hwy']) / 2
mpg['total']

sum(mpg['total']) / len(mpg['total'])
mpg['total'].mean()

mpg['total'].describe()
mpg['total'].plot.hist()
import matplotlib.pyplot as plt
plt.show()
plt.clf()
import numpy as np
mpg['test'] = np.where(mpg['total'] >= 20, 'pass', 'fail')
mpg.head()
mpg['test'].value_counts()
count_test = mpg['test'].value_counts()
count_test.plot.bar(rot=0)
plt.show()

mpg['grade'] = np.where(mpg['total'] >= 30, 'A',
np.where(mpg['total']>= 20, 'B', 'C'))
count_grade = mpg['grade'].value_counts()
count_grade
count_grade.plot.bar(rot=0)
plt.show()

count_grade = mpg['grade'].value_counts()
count_grade
count_grade.plot.bar(rot=0)
plt.show()
count_grade = mpg['grade'].value_counts().sort_index()
count_grade
count_grade.plot.bar(rot=0)
plt.show()
 
mpg['size'] = np.where((mpg['category'] == 'compact') |
                        (mpg['category'] == 'subcompact') |
                        (mpg['category'] == '2seater'), 'small', 'large')
        

