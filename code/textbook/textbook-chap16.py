import pandas as pd

df_raw=pd.read_csv('./data/exam.csv')
df=df_raw.head(10)
df
df['nclass']==1
df[(df['nclass']<50) | (df['english']<50)]

df.math

df[df['nclass']==1]['math']

df=pd.DataFrame({
    'var1':[1,2,3],
    'var2':[4,5,6]},
    index=['kim', 'lee', 'park'])

df.loc['kim']

df.loc[['kim', 'lee'], 'var2']
df.loc[:, 'var1']

df=df_raw.copy()
df
df.loc[[0, 3, 5]]
# :을 기준으로 앞에는 시작하는 행, 뒤에는 끝나는 행의 인덱스 입력
df.loc[:3]
df.loc[7:9]
df.loc[df['nclass']==1]






















