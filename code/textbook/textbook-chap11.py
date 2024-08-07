import json
import matplotlib.pyplot as plt
import pandas as pd

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
====================================================================================
## 0807
## 11-2 서울시 동별 외국인 인구 단계 구분도 만들기
import json
geo_seoul=json.load(open('./bigfile/SIG_Seoul.geojson', encoding='UTF-8'))

# 데이터 탐색
type(geo_seoul)
len(geo_seoul)
geo_seoul.keys()
geo_seoul["features"][0]
len(geo_seoul["features"])
len(geo_seoul["features"][0]) # 3
geo_seoul["features"][0].keys() # 3
geo_seoul["features"][0]["properties"] # 숫자에 따라 구 달라짐
geo_seoul["features"][0]["geometry"]

# 첫 번째 행정 구역의 기하학적 형상의 좌표 리스트를 coordinate_list 변수에 할당
coordinate_list=geo_seoul["features"][0]["geometry"]["coordinates"] # coordinates 위치 형상
len(coordinate_list[0][0])
coordinate_list[0][0] # [경도, 위도]

====================================================================================
# 종로구 경계선 표현 
import numpy as np
coordinate_array=np.array(coordinate_list[0][0])
x=coordinate_array[:,0]
y=coordinate_array[:,1]

import matplotlib.pyplot as plt
plt.plot(x,y)
plt.show()
plt.clf()

====================================================================================
# 함수로 만들기
# num은 서울시의 특정 행정 구역을 나타내는 인덱스
def draw_seoul(num):
    gu_name=geo_seoul['features'][num]['properties']['SIG_KOR_NM']
    coordinate_list=geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array=np.array(coordinate_list[0][0])
    x=coordinate_array[:,0]
    y=coordinate_array[:,1]

    plt.rcParams.update({'font.family':'Malgun Gothic'})
    plt.plot(x,y)
    plt.title(gu_name)
    plt.axis('equal') # 축 비율 1:1로 설정
    plt.show()
    plt.clf()
    
    return None

draw_seoul(23)

====================================================================================
# 서울시 전체 지도를 그리고 구별로 색깔을 다르게 하는 법
# 칼럼이 3개인 (gu_name, x, y) 인 데이터프레임으로 만들기

# plt.plot(s, y, hue='gu_name')

gu_name | x | y
종로구  |126|35
종로구  |126|35
종로구  |126|35
종로구  |126|35
종로구  |126|35
종로구  |126|35
...
중구    |126|35
중구    |126|35

# 1번
data = []

# geo_seoul의 모든 구에 대해 반복
for i in range(len(geo_seoul['features'])):
    # 좌표 및 구 이름 추출
    coordinate_list_all = geo_seoul['features'][i]['geometry']['coordinates']
    coordinate_array = np.array(coordinate_list_all[0][0])
    gu_name = geo_seoul['features'][i]['properties']['SIG_KOR_NM']
    
    # 좌표와 구 이름을 데이터 리스트에 추가
    for coord in coordinate_array:
        data.append({'gu_name': gu_name, 'x': coord[0], 'y': coord[1]})
df = pd.DataFrame(data)

plt.plot(x, y)
plt.show()
plt.clf()

# 2번
data = []
for i in range(25):
    xy = geo_seoul["features"][i]["geometry"]["coordinates"][0][0]
    for k in xy:
        data.append([geo_seoul["features"][i]["properties"]["SIG_KOR_NM"], k[0], k[1]])

df = pd.DataFrame(data, columns=['SIG_KOR_NM', 'x', 'y'])

====================================================================================
## 수업 코드 3번

# 구 이름 만들기
gu_name = geo_seoul['features'][0]['properties']['SIG_KOR_NM']
# 방법1
gu_name=list()
for i in range(25):
    gu_name.append(geo_seoul['features'][i]['properties']['SIG_KOR_NM'])
# 방법2
gu_name=[geo_seoul['features'][i]['properties']['SIG_KOR_NM'] for i in range(25)]

# x,y 판다스 데이터 프레임
def make_seouldf(num):
    gu_name=geo_seoul['features'][num]['properties']['SIG_KOR_NM']
    coordinate_list=geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array=np.array(coordinate_list[0][0])
    x=coordinate_array[:,0]
    y=coordinate_array[:,1]
    
    return pd.DataFrame({'gu_name': gu_name, 'x':x, 'y':y})


make_seouldf(23)

result=pd.DataFrame({})
for i in range(25):
    result=pd.concat([result, make_seouldf(i)], ignore_index=True)
    
result

====================================================================================
# 지도 그리기 시각화
import seaborn as sns

# 방법 1
gangnam_df=result.assign(is_gangnam=np.where(result['gu_name']=='강남구', '강남', '안강남'))
sns.scatterplot(
    data=gangnam_df,
    x='x', y='y',legend=False, palette=['grey', 'red'],
    hue='is_gangnam', s=2)
plt.show()
plt.clf()
# gangnam_df['is_gangnam'].unique() # unique 순서대로 palette 할당

# 방법 2
result=result.assign(color=np.where(result['gu_name'] == '강남구', 'red', 'grey'))
result
sns.scatterplot(data=result, x='x', y='y', hue='color', s=1)
plt.show()
plt.clf()

# 방법 3
gangnam_df=result.assign(is_gangnam=np.where(result['gu_name']!='강남구', '안강남', '강남'))
sns.scatterplot(
    data=gangnam_df,
    x='x', y='y', legend=False, 
    palette={"안강남": "grey", "강남": "red"},
    hue='is_gangnam', s=2)
plt.show()
plt.clf()

# palette='viridis'색상 지정하는 법
sns.scatterplot(data=result, x='x', y='y', hue='gu_name', legend=False, palette='viridis', s=2)
plt.show()
plt.clf()

# Q. 집값을 구별로 평균내서 평균값보다 해당 구역 집값이 높으면 red, 평균보다 낮으면 blue

====================================================================================
# 6교시 
geo_seoul=json.load(open('./bigfile/SIG_Seoul.geojson', encoding='UTF-8'))
geo_seoul['features'][0]['properties']

df_pop=pd.read_csv('data/Population_SIG.csv')
df_seoul_pop=df_pop.iloc[1:26] # 서울만 뽑기
df_seoul_pop['code']=df_seoul_pop['code'].astype(str)

import folium

center_x=result['x'].mean()
center_y=result['y'].mean()
# 밑바탕
map_sig=folium.Map(location=[37.5518099712906, 126.97315486480478],
            zoom_start=12,                  
            tiles='CartoDB positron')
map_sig.save('map_seoul.html')

# 코로플릿 Choropleth
bins=df_seoul_pop['pop'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1])
geo_seoul['features'][0]['properties']['SIG_CD']
folium.Choropleth(
    geo_data=geo_seoul,
    data=df_seoul_pop,
    columns=('code', 'pop'),
    bins=bins,
    fill_color='viridis',
    key_on='feature.properties.SIG_CD').add_to(map_sig) # key_on이 도대체 뭔데 ..
map_sig.save('map_seoul.html')


# 점 찍는 법
# 종로구의 중앙을 찾아서
make_seouldf(0).iloc[:,1:3].mean()

folium.Marker([37.583744, 126.983800], popup='종로구').add_to(map_sig)
map_sig.save('map_seoul.html')

====================================================================================
## houseprice 위치 위도 경도 찍어서 지도에 표시하기
# 각 집들을 marker로 찍어라
# Longitude 경도 Latitude 위도

house_df = pd.read_csv("./data/houseprice-with-lonlat.csv")

house_df = house_df[["Longitude", "Latitude"]]

center_x=house_df["Longitude"].mean()
center_y=house_df["Latitude"].mean()

map_sig=folium.Map(location = [42.034, -93.642],
                  zoom_start = 12,
                  tiles="cartodbpositron")

from folium.plugins import MarkerCluster
marker_cluster = MarkerCluster().add_to(map_sig)

# 방법 1
for idx, row in house_df.iterrows():
    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup="집들,,"
    ).add_to(marker_cluster)

map_sig.save("map_ames.html")

# 방법 2
for i in range(len(house_df)):
    folium.Marker(
        location=[house_df.iloc[i,1], house_df.iloc[i,0]],
        popup="houses,,"
    ).add_to(marker_cluster)
map_sig.save("map_ames.html")

# 점 하나하나 찍는 코드
for lat, lon in zip(house_df['Latitude'], house_df['Longitude']):
  folium.Marker([lat, lon]).add_to(my_map)
  
my_map.save("map_seoul2.html")

====================================================================================
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

====================================================================================
