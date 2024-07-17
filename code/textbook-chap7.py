import pandas as pd
import numpy as np

df = pd.DataFrame({'sex'  : ['M', 'F', np.nan, 'M', 'F'],
                   'score': [5, 4, 3, 4, np.nan]})
pd.isna(df).sum()
df['score'] + 1

# 결측치 제거하기
df.dropna()                          # 모든 변수 결측치 제거
df.dropna(subset = 'score')          # score 변수에서 결측치 제거
df.dropna(subset = ['score', 'sex']) # 여러 변수 결측치 제거법

exam = pd.read_csv('data/exam.csv')

# 데이터 프레임 location을 사용한 인덱싱
# exam.loc[행 인덱스, 열 인덱스]
exam.loc[0, 0]
exam.iloc[0:2, 0:4]
# exam.loc[[2, 7, 4], ['math']] = np.nan
exam.iloc[[2, 7, 4], 2] = np.nan
exam.iloc[[2, 7, 4], 2] = 3
exam

# 수학 점수가 50점 이하인 학생들 점수를 다 50점으로 상향 조정
exam.loc[exam['math'] <= 50, ['math']] = 50

# 영어 점수가 90점 이상은 90점으로 하향 조정 (iloc 사용)
exam.loc[exam['english'] >= 90, ['english']] = 90

# iloc을 사용해서 조회하려면 무조건 숫자 벡터가 들어가야 함
exam.iloc[exam['englsih'] >= 90, 3] # 실행 안됨
exam.iloc[np.where(exam['english']>=90),3] # 실행 됨
exam.iloc[np.where(exam['english']>=90)[0],3] # np.where도 튜플이라 [0] 꺼내오면 됨
exam.iloc[exam[exam['english'] >= 90].index, 3] # index 벡터도 작동
exam

# math 점수 50 이하 "-" 변경
exam.loc[exam['math'] <= 50, 'math'] = '-'

# '-' 결측치를 수학점수 평균 바꾸고 싶은 경우
# 1번 (가장 좋은듯)
math_mean = exam.loc[(exam['math'] != '-'), 'math'].mean()
exam.loc[exam['math'] == '-', 'math'] = math_mean

# 2번 (이것도)
math_mean = exam[exam['math'] != '-']['math'].mean()
exam.loc[exam['math']== '-', 'math'] = math_mean

# 3번
exam.loc[exam['math'] == '-', ['math']] = np.nan
math_mean = exam['math'].mean()
exam.loc[pd.isna(exam['math']), ['math']] = math_mean

# 4번 (교재)
math_mean = exam[exam['math'] != '-']['math'].mean()
exam['math'] = exam['math'].replace('-', math_mean)
exam

# 5번
math_mean = exam.query('math not in ["-"]')['math'].mean()
exam.loc[exam['math']=='-','math'] = math_mean
