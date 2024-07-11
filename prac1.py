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
