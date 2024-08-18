import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 데이터 불러오기
df = pd.read_csv('lsbigdata-project1/data/Common_Jobs_by_Major.csv')

# 필요한 열만 선택
df = df[['Detailed Occupation', 'Total Population', 'ID CIP2']]

# CIP2 코드 숫자를 한글 전공명으로 변경
df['ID CIP2'] = df['ID CIP2'].replace(14, '공학')
df['ID CIP2'] = df['ID CIP2'].replace(19, '인문과학')
df['ID CIP2'] = df['ID CIP2'].replace(27, '수학_통계')
df['ID CIP2'] = df['ID CIP2'].replace(31, '공원_레크리에이션_레저')
df['ID CIP2'] = df['ID CIP2'].replace(26, '생물학')

# Detailed Occupation 변수를 한글 직업군으로 변경
df['Detailed Occupation'] = df['Detailed Occupation'].replace('Other managers', '기타경영자')
df['Detailed Occupation'] = df['Detailed Occupation'].replace('Software developers', '소프트웨어개발자')
df['Detailed Occupation'] = df['Detailed Occupation'].replace('Civil engineers', '토목기사')
df['Detailed Occupation'] = df['Detailed Occupation'].replace('Miscellaneous engineers, including nuclear engineers', '기타기술자')
df['Detailed Occupation'] = df['Detailed Occupation'].replace('Physicians', '의료인')

# 전공별로 필터링
df_engineering = df[df['ID CIP2'] == '공학']
df_human_sciences = df[df['ID CIP2'] == '인문과학']
df_math_statistics = df[df['ID CIP2'] == '수학_통계']
df_parks_recreation_leisure = df[df['ID CIP2'] == '공원_레크리에이션_레저']
df_biology = df[df['ID CIP2'] == '생물학']

# 1번. 공학 전공별 직업분포확인
df_pie = df_engineering.groupby('Detailed Occupation').sum('Total Population').reset_index()[['Detailed Occupation', 'Total Population']]

# 파이 차트 생성
fig = go.Figure()

# 파이 차트에 트레이스 추가
fig.add_trace(go.Pie(
    values=df_pie['Total Population'], 
    labels=df_pie['Detailed Occupation'],
    direction='clockwise', 
    sort=True
))

# 레이아웃 업데이트
fig.update_layout(title=dict(text='공학 전공자별 직업 분포', x=0.5))

# 차트 출력
fig.show()