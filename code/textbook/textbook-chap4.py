import pandas as pd
import numpy as np
df = pd.DataFrame({'name': ['김지훈', '이유진', '박동현', '김민지'],
'english': [90, 80, 60, 70],
'math': [50, 60, 100, 20]})

df
df['name']

type(df)
type(df['name'])

sum(df['english']) / 4

# p.84 혼자서 해보기
df = pd.DataFrame({'제품': ['사과', '딸기', '수박'],
'가격': [1800, 1500, 3000],
'판매량': [24, 38, 13]})
df

sum(df['가격']) / len(df['가격'])
sum(df['판매량']) / len(df['판매량'])

! pip install openpyxl
import pandas as pd
import numpy as np

df_exam = pd.read_excel('data/excel_exam.xlsx')
df_exam

sum(df_exam['math'])/20
sum(df_exam['english'])/20
sum(df_exam['science'])/20

df_exam.shape
len(df_exam)
df_exam.size

df_exam = pd.read_excel('data/excel_exam.xlsx')
df_exam

df_exam['total'] = df_exam['math'] + df_exam['math'] + df_exam['science']
df_exam['mean'] = (df_exam['total']) / len(df_exam['total'])

df_exam[df_exam['math']>50]

# 문제
df_exam [(df_exam['math']>50) & (df_exam['english']>50)]

# 수학은 평균보다 높지만 영어는 평균보다 낮은 
mean_m = np.mean(df_exam['math'])
mean_e = np.mean(df_exam['english'])

df_exam[(df_exam['math'] > mean_m) &
(df_exam['english'] < mean_e)]

df_nc3 = df_exam [df_exam['nclass'] == 3]
df_nc3[['math', 'english', 'science']]
df_nc3[1:]
df_nc3[0:1]

df_exam[0:10:2]
df_exam[7:16]

df_exam.sort_values('math', ascending = False)
df_exam.sort_values(['nclass', 'math'], ascending = [True, False])

np.where(a>3, 'Up', 'Down')
df_exam['updown'] = np.where(df_exam['math']>50, 'Up', 'Down')
df_exam







