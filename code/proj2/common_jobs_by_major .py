---
title: '전공별 직업 분포 확인'
author: "Jinii"
format: 
  dashboard:
    logo: "penguins-cover.png"
    nav-buttons:
      - icon: github
        href: https://github.com/Jinii1/lsbigdata-project1
---

# {.toolbar}

여기는 사이드바에 대한 내용이 들어갈 곳입니다.

# Page 1

## 칼럼

### 첫번째 {height=60%}

```{python}
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
df=pd.read_csv('../../data/Common Jobs by Major.csv')
df['Detailed Occupation'].value_counts(ascending=True)



fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species",
    # trendline="ols" # p.134
)
fig.show()
```

### 두번째 {height=40%}

#### 테스트1 {.tabset}

```{python}
from itables import show
show(penguins, buttons = ['copy', 'excel', 'pdf'])
```


::: {.card title="My Title"}

카드안에 들어있는 텍스트 입니다.

![팔머펭귄](penguins-cover.png)

:::

# Page 2

```{python}
articles = 100
comments = 50
spam_num = 300
```

## Row 

```{python}
#| content: valuebox
#| title: "Articles per day"
#| icon: pencil
#| color: primary
dict(
  value = articles
)
```

```{python}
#| content: valuebox
#| title: "Comments per day"
dict(
  icon = "chat",
  color = "primary",
  value = comments
)
```

```{python}
#| content: valuebox
#| title: "Spam per day"
dict(
  icon = "airplane-engines",
  color = "#f0330b",
  value = spam_num
)
```

## Row  

```{python}
import pandas as pd
import numpy as np
import plotly.express as px
from palmerpenguins import load_penguins

penguins = load_penguins()
# penguins.head()

fig = px.scatter(
    penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    color="species",
    # trendline="ols" # p.134
)
fig.show()
```




fig=go.figure()
fig.add_trace(go.Bar(
  x=CIP,
  y=Detailed Occcupation['Other managers'],
  name='Other managers'))

fig.add_trace(go.Bar(
  x=CIP,
  y=Detailed Occcupation['Software developers'],
  name='Software developers'))

fig.add_trace(go.Bar(
  x=CIP.index,
  y=Detailed Occcupation['Physicians'],
  name='Physicians'))

fig.add_trace(go.Bar(
  x=CIP.index,
  y=Detailed Occcupation['Civil engineers'],
  name='Civil engineers'))

fig.add_trace(go.Bar(
  x=CIP.index,
  y=Detailed Occcupation['Miscellaneous engineers, including nuclear engineers'],
  name='Miscellaneous engineers, including nuclear engineers'))

fig.update_layout(barmode='group', bargroupgap=0.2,
                  title:'전공별 직업분포 확인',
                  title_x=0.5, margin=margins_P)

fig.show()