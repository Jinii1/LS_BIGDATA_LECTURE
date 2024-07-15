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

df_exam = pd.read_excel('data/excel_exam.xlsx')
df_exam

sum(df_exam['math'])/20
sum(df_exam['english'])/20
sum(df_exam['science'])/20

df_exam.shape
len(df_exam)
df_exam.size

df_exam = pd.read_excel('data/excel_exam.xlsx', sheet_name ='Sheet2')
df_exam

df_exam['total'] = df_exam['math'] + df_exam['math'] + df_exam['science']
df_exam['mean'] = (df_exam['total']) / len(df_exam['total'])

df_exam[df_exam['math']>50]

df_exam [(df_exam['math']>50) & (df_exam['english']>50)]















