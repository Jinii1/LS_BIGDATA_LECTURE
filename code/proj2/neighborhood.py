import numpy as np
import pandas as pd

df = pd.read_csv("./data/houseprice-with-lonlat.csv")
df.info()

df['Neighborhood'].describe()
df['Neighborhood'].isna().sum()

import json
geo = json.load(open('./data/us-states.json', encoding='UTF-8'))

import folium
m = folium.Map(location = [42.054035,-93.619754], zoom_start = 5)
m.save('./data/map1.html')
