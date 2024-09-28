# 학교에서 교사 한 명당 맡은 학생 수가 가장 많은 학교를 찾고,
# 그 학교의 전체 교수를 구하시오 (정수 출력)

import pandas as pd
df = pd.read_csv('../data/data6-1-2.csv')

df['전체'] = df.iloc[:, 2:].sum(axis=1)
df['전체/교사'] = df['전체'] - df['교사수']
df['전체/교사'].idxmax()
int(df.loc[7, '교사수'])

df['1인교사수'] = (df['1학년']+df['2학년']+df['3학년']+df['4학년']+df['5학년']+df['6학년'])/df['교사수']
df=df.sort_values('1인교사수', ascending=False)
df.iloc[0,1]