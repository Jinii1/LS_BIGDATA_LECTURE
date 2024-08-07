import pandas as pd
mpg=pd.read_csv('data/mpg.csv')

# 산점도 만들기
import plotly.express as px
fig=px.scatter(data_frame=mpg, x='cty', y='hwy', color='drv',
                width=600, height=400)   #그래프 가로 크기 width, 세로 크기 height
fig.show()

# 자동차 종류별 빈도 구하기 - 막대 그래프
df = mpg.groupby('category', as_index=False) \
        .agg(n=('category', 'count'))
df
fig=px.bar(data_frame=df, x='category', y='n', color='category')
fig.show()


# 선 그래프
economics=pd.read_csv('data/economics.csv')
fig=px.line(data_frame=economics, x='date', y='psavert')
fig.show()

# 상자 그림
fig=px.box(data_frame=mpg, x='drv', y='hwy', color='drv')
fig.show()
fig.write_html('box.html')

# 새 창에 그래프 출력하기
import plotly
plotly.io.renderers.default='browser'
# 설정 원래대로 돌리기
plotly.io.renderers.default='jupyterlab'
