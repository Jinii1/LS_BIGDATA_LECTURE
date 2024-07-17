import pandas as pd
import numpy as np

# 데이터 전처리 함수
# query()
# df[]
# sort_values()
# groupby()
# assign()
# agg()
# merge()
# concat()

exam = pd.read_csv('data/exam.csv')
exam.query('nclass == 1')
exam.query('nclass == 2')
exam.query('nclass != 1')
exam.query('nclass != 3')
exam.query('math > 50')
exam.query('math < 50')
exam.query('english >= 50')
exam.query('english <= 80')
exam.query('nclass == 1 & math >= 50')
exam.query('nclass == 2 and english >= 80')
exam.query('math >= 90 | english >= 90')
exam.query('english < 90 or science < 50')
exam.query('nclass in [1, 3, 5]')
exam.query('nclass not in [1, 3, 5]')

exam[~exam['nclass'].isin([1, 2])]

exam[['id', 'nclass']]
exam[['nclass']]
exam.drop(columns = 'math')
exam.drop(columns = ['math', 'english'])
exam.query('nclass == 1')\
        [['english', 'math']]\
        .head()
        
# p. 150
mpg_new = mpg[['category', 'cty']]
mpg_new.query('category == "suv"')['cty'].mean()
mpg_new.query('category == "compact"')['cty'].mean()
        
# 정렬하기
exam.sort_values(['nclass', 'english'], ascending = [True, False])

# 변수추가
exam = exam.assign(
    total = exam['math'] +exam['english'] + exam['science'],
    mean = (exam['math'] +exam['english'] + exam['science']) / 3) \
    .sort_values('total', ascending = False).head(5)
    
exam.assign(test = np.where(exam['science']>=60, 'pass', 'fail'))

# lambda 함수 사용하기
exam2 = pd.read_csv('data/exam.csv')
exam2 = exam.assign(
    total = lambda x: x['math'] + x['english'] + x['science'],
    mean = lambda x: x['total'] / 3) \
    .sort_values('total', ascending = False)
    
# 그룹을 나눠 요약을 하는 .groupby() + .agg() 콤보
exam2.agg(mean_math = ('math', 'mean'))
exam2.groupby('nclass') \
     .agg(mean_math = ('math', 'mean'))
     
# 반 별 각 과목별 평균
exam2.groupby('nclass')\
     .agg(mean_math = ('math', 'mean'),\
          mean_english = ('english', 'mean'),
          mean_science = ('science', 'mean'))

mpg = pd.read_csv('C:/Users/USER/Downloads/mpg.csv')
mpg.assign(total = (mpg['hwy'] + mpg ['cty']) / 2) \
   .groupby('manufacturer') \
   .agg(mean_tot = ('total', 'mean')) \
   .sort_values('mean_tot', ascending = False) \
   .head()

exam.agg(mean_math = ('math', 'mean'))
exam.groupby('nclass', as_index = False)\
    .agg(mean_math = ('math', 'mean'))
exam.groupby('nclass')\
    .agg(mean_math = ('math', 'mean'),
         sum_math = ('math', 'sum'),
         median_math = ('math', 'median'),
         n = ('nclass', 'count'))
exam.groupby('nclass').mean()

mpg.groupby(['manufacturer', 'drv'])\
    .agg(mean_cty = ('cty', 'mean'))
mpg.query('manufacturer == "audi"')\
    .groupby(['drv'])\
    .agg(n = ('drv', 'count'))
mpg.query('manufacturer == "chevrolet"')\
    .groupby(['drv'])\
    .agg(n = ('drv', 'count'))
mpg['drv'].value_counts()

mpg.query('category == "suv"')\
    .assign(total = (mpg['hwy'] + mpg['cty']) / 2)\
    .groupby('manufacturer')\
    .agg(mean_tot = ('total', 'mean'))\
    .sort_values('mean_tot', ascending = False).head()

# p. 166
# 1번
mpg.groupby('category')\
    .agg(mean_cty= ('cty', 'mean'))
    
# 2번
mpg.groupby('category')\
    .agg(mean_cty= ('cty', 'mean')).sort_values('mean_cty', ascending = False)
    
# 3번
mpg.groupby('manufacturer')\
    .agg(mean_hwy = ('hwy', 'mean'))\
    .sort_values('mean_hwy', ascending = False)\
    .head(3)

# 4번
mpg.query('category == "compact"')\
    .groupby('manufacturer')\
    .agg(n = ('category', 'count'))\
    .sort_values('n', ascending = False)

test1 = pd.DataFrame({'id': [1, 2, 3, 4, 5],
                      'midterm': [60, 80, 70, 90, 85]})
test2 = pd.DataFrame({'id': [1, 2, 3, 4, 5],
                      'midterm': [70, 83, 65, 95, 80]})
total = pd.merge(test1, test2, how='left', on = 'id')

group_a = pd.DataFrame({'id': [1, 2, 3, 4, 5],
                        'test': [60, 80, 70, 90, 85]})
group_b = pd.DataFrame({'id': [6, 7, 8, 9, 10],
                        'test': [70, 83, 65, 95, 80]})
group_all = pd.concat([group_a, group_b])

# p. 173
fuel = pd.DataFrame({'fl'     : ['c', 'd', 'e', 'p', 'r'],
                    'price_f1': [2.35, 2.38, 2.11, 2.76, 2.22]})
                    
# 1번
mpg = pd.merge(mpg, fuel, how = 'left', on = 'fl')

# 2번
mpg[['model', 'fl', 'price_f1']].head()

# 정리하기

# 조건 맞는 데이터 추출
exam.query('english <= 80')
exam.query('nclass == 1 & math >= 50')
exam.query('math >= 90 | english >= 90')
exam.query('nclass in [1, 3, 5]')

# 필요한 변수 추출
exam['math']
exam[['math', 'english']]
exam.drop(columns = 'math')
exam.drop(columns = ['math', 'english'])

# pandas 명령어 조합
exam.query('math >= 50')[['id', 'math']].head()

# 순서대로 정렬
exam.sort_values('math', ascending = False) # 내림차순
exam.sort_values(['nclass', 'math'], ascending = [True, False])

# 파생변수 추가
exam.assign(total = exam['math'] + exam['english'] + exam['science'],
            mean = (exam['math'] + exam['english'] + exam['science']) / 3)
exam.assign(test = np.where(exam['science'] >= 60, 'pass', 'fail'))
exam.assign(total = exam['math'] + exam['english'] + exam['science'])\
    .sort_values('total')\
    .head()

# 집단별 요약
exam.groupby('nclass')\
    .agg(mean_math = ('math', 'mean'))

mpg.groupby(['manufacturer', 'drv'])\
    .agg(mean_cty = ('cty', 'mean'))

pd.merge(test1, test2, how = 'left', on = 'id')
pd.concat([group_a, group_b])

# p. 176
midwest = pd.read_csv('data/midwest.csv')

# 1번
midwest.assign(ratio = ((midwest['poptotal'] - midwest['popadults']) / midwest['poptotal']) * 100)

# 2번
midwest.sort_values('ratio', ascending = False)\
       .groupby('county')\
       .head(5)

test1 = pd.DataFrame({'id'     : [1, 2, 3, 4, 5],
                      'midterm': [60, 80, 70, 90, 85]})
test2 = pd.DataFrame({'id'   : [1, 2, 3, 4, 5],
                      'final': [70, 83, 65, 95, 80]})
total = pd.merge(test1, test2, how = 'left', on = 'id')
total = pd.merge(test2, test1, how = 'right', on = 'id')
# inner join 교집합으로 가지고 있는 것만
total = pd.merge(test1, test2, how = 'inner', on = 'id')
# outer join 합집합 (모두 다)
total = pd.merge(test1, test2, how = 'outer', on = 'id')

name = pd.DataFrame({'nclass': [1, 2, 3, 4, 5],
                    'teacher': ['kim', 'lee', 'park', 'choi', 'jung']})
exam_new = pd.merge (exam, name, how = 'left', on = 'nclass')

score1 = pd.DataFrame({'id'     : [1, 2, 3, 4, 5],
                      'score': [60, 80, 70, 90, 85]})
score2 = pd.DataFrame({'id'   : [6, 7, 8, 9, 10],
                      'score': [70, 83, 65, 95, 80]})
score_all = pd.concat([score1, score2])
