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
