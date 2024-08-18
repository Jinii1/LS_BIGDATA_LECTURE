# ! pip install palmerpenguins

import plotly.express as px
import plotly.graph_objects as go
from palmerpenguins import load_penguins
import pandas as pd
import numpy as np

penguins=load_penguins()
penguins.head()
penguins.columns
penguins['species'].unique()
# x: bill_length_mm 
# y: bill_depth_mm  
fig=px.scatter(
    penguins,
    x='bill_length_mm',
    y='bill_depth_mm',
    color='species',
    # trendline='ols' # p.134
)
fig.show()

# 레이아웃 업데이트
fig.update_layout(
    title=dict(text="펭귄 종별 부리 길이 vs. 깊이", font=dict(color="white")),
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white"),
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    legend=dict(font=dict(color="white")),
)
fig.show()

# 같은 코드를 dict() 말고 {} 중괄호 이용했을때
fig.update_layout(
    title={"text": "펭귄 종별 부리 길이 vs. 깊이", "font": {"color": "white"}},
    paper_bgcolor="black",
    plot_bgcolor="black",
    font={"color": "white"},
    xaxis={
        "title": {"text": "부리 길이 (mm)", "font": {"color": "white"}}, 
        "tickfont": {"color": "white"},
        "gridcolor": 'rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    },
    yaxis={
        "title": {"text": "부리 깊이 (mm)", "font": {"color": "white"}}, 
        "tickfont": {"color": "white"},
        "gridcolor": 'rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    },
    legend={"font": {"color": "white"}},
)

========================================================================================
# 제목 크기 키우고 점 크기 키우고 범례 제목을 펭귄 종으로 변경

# 산점도 생성
fig = px.scatter(
    penguins,
    x='bill_length_mm',
    y='bill_depth_mm',
    color='species',
    color_discrete_map={    # 각 범주에 대해 특정 색상을 지정하는 역할
        'Adelie': 'pink',
        'Chinstrap': 'red',
        'Gentoo': 'skyblue'
    }
)

# 레이아웃 업데이트
fig.update_layout(
    title=dict(
        text="펭귄 종별 부리 길이 vs. 깊이", 
        font=dict(size=24, color="white")  # 타이틀 크기 키우기
    ),
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white"),
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    legend=dict(
        title=dict(text="펭귄 종", font=dict(color="white")),  # 범례 제목 변경
        font=dict(color="white")
    )
)

# 점 크기 및 투명도 업데이트
fig.update_traces(marker=dict(size=10, opacity=0.7))  # 점 크기 키우기 및 투명도 조정
fig.show()

# update_traces() 기존의 모든 트레이스를 일괄적으로 수정
# add_trace() 새로운 트레이스를 추가, data 속성값들을 딕셔너리로 구성하여 할당
# 각각의 딕셔너리는 먼저 type 속성을 사용하여 어떤 타입의 트레이스인지 설정
# -> 해당 트레이스에 관련된 data 속성 설정

========================================================================================
from sklearn.linear_model import LinearRegression

model=LinearRegression()
penguins=penguins.dropna()
x=penguins[['bill_length_mm']]
y=penguins['bill_depth_mm']

model.fit(x, y) # x와 y 데이터를 사용하여 선형 회귀 모델을 학습
model.coef_
model.intercept_

linear_fit = model.predict(x)
# 학습된 모델을 사용하여 x에 대한 예측값을 계산, 이 예측값은 회귀 직선 상의 y 값들을 의미

fig.show()

fig.add_trace(
    go.Scatter(
        mode='lines',
        x=penguins['bill_length_mm'], y=linear_fit,
        name='선형회귀직선',
        line=dict(dash='dot',color='white')
    )
)

fig.show()

========================================================================================
# 범주형 변수로 회귀분석 진행하기
# 범주형 변수인 'species'를 더미 변수로 변환
# 더미 변수는 각 범주형 값에 대해 0 또는 1을 갖는 이진 변수
penguins_dummies = pd.get_dummies(penguins, 
                                  columns=['species'],
                                  drop_first=False) # 모든 범주에 대해 더미 변수 생성
penguins_dummies.columns
penguins_dummies.iloc[:,-3:]

# x와 y 설정
x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
y = penguins_dummies["bill_depth_mm"]

# 모델 학습
model = LinearRegression()
model.fit(x, y)

model.coef_
model.intercept_

regline_y=model.predict(x)

# 각각 회귀직선 그리기
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x=penguins["bill_length_mm"], y=y, 
                hue=penguins["species"], palette="deep",
                legend=False)
sns.scatterplot(x=penguins["bill_length_mm"], y=regline_y,
                color="black")
plt.show()
plt.clf()

========================================================================================
y=ax1+bx2+c

y=model.coef[0]*x[0]+model.coef[1]*x[1]+model.coef[2]*x[2]+model.intercept_

# y = 0.2 * bill_length -1.93 * species_Chinstrap -5.1 * species_Gentoo + 10.56
# 이런 회귀직선을 우리가 그린거다
# penguins
# species    island  bill_length_mm  ...  body_mass_g     sex  year
# Adelie     Torgersen            39.5  ...       3800.0  female  2007
# Chinstrap  Torgersen            40.5  ...       3800.0  female  2007
# Gentoo     Torgersen            40.5  ...       3800.0  female  2007
# x1, x2, x3
# 39.5, 0, 0
# 40.5, 1, 0
# y = 0.2 * bill_length -1.93 * species_Chinstrap -5.1 * species_Gentoo + 10.56
0.2 * 40.5 -1.93 * True -5.1* False + 10.56