! pip install plotly
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# 데이터 불러오기
df_covid19_100=pd.read_csv('../../data/df_covid19_100.csv')
df_covid19_100_wide=pd.read_csv('./data/df_covid19_100_wide.csv')
df_covid19_stat=pd.read_csv('./data/df_covid19_stat.csv')
df_covid19_100.info()

# 전처리
margins_P = {'t': 50, 'b': 25, 'l': 25, 'r': 25} # 여백 설정을 위한 변수 설정

# data, layout, frame 세 가지 속성
# Figure() 사용하여 Plotly 객체 초기화하여 생성
# Plotly 초기화 함수로 data 속성값 설정
fig=go.Figure(
    data=[{
        'type': 'scatter', 'mode': 'markers', # scatter 트레이스의 markers 속성 설정
        'x': df_covid19_100.loc[df_covid19_100['iso_code'] == 'KOR', 'date'],
        'y': df_covid19_100.loc[df_covid19_100['iso_code'] == 'KOR', 'new_cases'],
        'marker': {'color':'#264E86'}},
        {'type': 'scatter', 'mode': 'lines', # scatter 트레이스의 lines 속성 설정
        'x': df_covid19_100.loc[df_covid19_100['iso_code'] == 'KOR', 'date'],
        'y': df_covid19_100.loc[df_covid19_100['iso_code'] == 'KOR', 'new_cases'],
        'line': {'color': '#5E88FC', 'dash':'dash'}} # dash: 점선
    ],
    layout={
        'title':'코로나19 발생 현황',
        'xaxis': {'title':'날짜', 'showgrid':False},
        'yaxis':{'title':'확진자수'},
        'margin':margins_P
    }
    ).show() 

# data 속성은 데이터를 표현하는 트레이스를 구성하는 세부 속성을 의미
# 트레이스: 데이터를 시각화한 도형 레이어 (scatter, pie, bar ..)
# 트레이스를 추가하기 위해선 add_trace() 사용

# 트레이스는 특정 데이터 집합을 도형으로 표현하기 위해 필요한 data 속성의 집합
# 트레이스의 세부적인 특성을 설정하는 속성이 data 속성

==================================================================================
# 프레임속성을 이용한 애니메이션
# 시간에 따라 데이터를 추가하는 방식의 애니메이션 프레임 생성
frames = []
dates = df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"].unique()

for date in dates:
    frame_data = {
        "data": [
            {
                "type": "scatter",
                "mode": "markers",
                "x": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "date"],
                "y": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "new_cases"],
                "marker": {"color": "red"}
            },
            {
                "type": "scatter",
                "mode": "lines",
                "x": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "date"],
                "y": df_covid19_100.loc[(df_covid19_100["iso_code"] == "KOR") & (df_covid19_100["date"] <= date), "new_cases"],
                "line": {"color": "blue", "dash": "dash"}
            }
        ],
        "name": str(date)
    }
    frames.append(frame_data)

# x축과 y축의 범위 설정
x_range = ['2022-10-03', '2023-01-11']
y_range = [8900, 88172]

# 애니메이션을 위한 레이아웃 설정
margins_P = {"l": 25, "r": 25, "t": 50, "b": 50}
layout = {
    "title": "코로나 19 발생현황",
    "xaxis": {"title": "날짜", "showgrid": False, "range": x_range},
    "yaxis": {"title": "확진자수", "range": y_range},
    "margin": margins_P,
    "updatemenus": [{     # 애니메이션 컨트롤 버튼 (재생, 일시정지)
        "type": "buttons",
        "showactive": False,
        "buttons": [{
            "label": "Play",
            "method": "animate",
            "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
        }, {
            "label": "Pause",
            "method": "animate",
            "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
        }]
    }]
}

# Figure 생성
fig = go.Figure(
    data=[
        {
            "type": "scatter",
            "mode": "markers",
            "x": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"],
            "y": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "new_cases"],
            "marker": {"color": "red"}
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "date"],
            "y": df_covid19_100.loc[df_covid19_100["iso_code"] == "KOR", "new_cases"],
            "line": {"color": "blue", "dash": "dash"}
        }
    ],
    layout=layout,
    frames=frames
)

fig.show()

# import os
# cwd=os.getcwd()
# cwd