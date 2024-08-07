import json
import matplotlib.pyplot as plt

# 단계 구분도: 지역별 통게치를 색깔 차이로 표현한 지도

# 지역별 위도, 경도 좌표가 있는 지도 데이터 필요
geo=json.load(open('./bigfile/SIG.geojson', encoding='UTF-8'))
geo['features'][0]['properties']
geo['features'][0]['geometry']

# 시군구별 인구 통계데이터가 담긴 csv 파일
import numpy as np
df_pop=pd.read_csv('./data/Population_SIG.csv')
df_pop.info()
df_pop.head()

# df_pop: 행정 구역 코드를 나타냄, 문자 타입으로 있어야 지도 만드는데 활용할 수 있어서 타입 변경
df_pop['code']=df_pop['code'].astype(str)

# 배경 지도 만들기
import folium
folium.Map(location=[35.95, 127.7], # 지도의 중심 위도, 경도 좌표 입력
            zoom_start=8)           # 지도를 확대할 정도를 입력
map_sig=folium.Map(location=[35.95, 127.7], 
                   zoom_start=8,
                   tiles='Cartodbpositron') # 지도 종류
    
# 단계구분도 만들기
folium.Choropleth(
    geo_data=geo,                           # 지도 데이터
    data=df_pop,                            # 통계데이터
    columns=('code', 'pop'),                # df_pop 행정 구역 코드, 인구
    key_on='feature.properties.SIG_CD') \   # geo 행정구역 코드
        .add_to(map_sig)
map_sig

# 계급 구간 정하기 (지역 색깔별로 표현하기 위해)
bins=list(df_pop['pop'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))
bins

# 디자인 수정하기
# 배경 지도 만들기
map_sig=folium.Map(location=[35.95, 127.7],
                   zoom_start=8,
                   tiles='cartodbpositron')
# 단계 구분도 만들기
folium.Choropleth(
        geo_data=geo,
        data=df_pop,
        columns=('code', 'pop'),
        key_on='feature.properties.SIG_CD',
        fill_color='YlGnBu',
        fill_opacity=1,
        line_opacity=0.5,
        bins=bins) \
        .add_to(map_sig)
# -> 인구가 많을수록 파란색, 적을수록 노란색에 가깝게 표현

## 11-2 서울시 동별 외국인 인구 단계 구분도 만들기
import json
geo_seoul=json.load(open('./data/EMD_Seoul.geojson', encoding='UTF-8'))
geo_seoul['features'][0]['properties']
geo_seoul['features'][0]['geometry']

foreigner=pd.read_csv('data/Foreigner_EMD_Seoul.csv')
foreigner.head()
foreigner.info()
foreigner['code']=foreigner['code'].astype(str)

bins=list(foreigner['pop'].quantile([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1]))
bins

# 배경 지도 만들기
map_seoul=folium.Map(location=[37.56, 127],
                     zoom_start=12,
                     tiles='cartodbpositron')

# 단계구분도 만들기
folium.Choropleth(
    geo_data=geo_seoul,
    data=foreigner,
    columns=('code', 'pop'),
    key_on='feature.properties.ADM_DR_CD',
    fill_color='Blues',
    nan_fill_color='White',
    fill_opacity=1,
    line_opacity=0.5,
    bins=bins) \
        .add_to(map_seoul)
map_seoul

geo_seoul_sig=json.load(open('data/SIG_Seoul.geojson', encoding='UTF-8'))

folium.Choropleth(geo_data=geo_seoul_sig,
                  fill_opacity=0,
                  line_weight=4) \
        .add_to(map_seoul)
map_seoul
map_seoul.save('map_seoul.html')
import webbrowser
webbrowser.open_new('map_seoul.html')
