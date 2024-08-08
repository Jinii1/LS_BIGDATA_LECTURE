# ! pip install palmerpenguins

import plotly.express as px
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

========================================================================================
# 제목 크기 키우고 점 크기 키우고 범례 제목을 펭귄 종으로 변경

# 산점도 생성
fig = px.scatter(
    penguins,
    x='bill_length_mm',
    y='bill_depth_mm',
    color='species',
    color_discrete_map={
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

========================================================================================
from sklearn.linear_model import LinearRegression

model=LinearRegression()
penguins=penguins.dropna()
x=penguins[['bill_length_mm']]
y=penguins['bill_depth_mm']

model.fit(x, y)
model.coef_
model.intercept_

linear_fit = model.predict(x)

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
penguins_dummies = pd.get_dummies(penguins, 
                                  columns=['species'],
                                  drop_first=False)
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

plt.plot(x['bill_length_mm'], regline_y)
plt.show()
plt.clf()

import seaborn as sns

# 회귀직선 그래프 만들기

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

========================================================================================























