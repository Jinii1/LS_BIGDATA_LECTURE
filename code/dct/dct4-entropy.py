# 펭귄 데이터 부리길이 예측 모형 만들기
# 엘라스틱 넷 & 디시젼트리 회귀모델 사용
# 모든 변수 자유롭게 사용!
# 종속변수: bill_length_mm

# 지금 하는 것 - 3개의 변수를 가지고, 트리모델을 만들고 싶은 경우
import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
from sklearn.preprocessing import OneHotEncoder

penguins = load_penguins()
penguins=penguins.dropna()

df_X=penguins.drop("species", axis=1)
df_X=df_X[["bill_length_mm", "bill_depth_mm"]]
y=penguins[['species']]


# 모델 생성
from sklearn.tree import DecisionTreeClassifier

## 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV
model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42)

param_grid={
    'max_depth': np.arange(7, 20, 1),
    'min_samples_split': np.arange(10, 30, 1)
}

grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5
)

grid_search.fit(df_X,y)

grid_search.best_params_ #8, 22
grid_search.cv_results_
grid_search.best_score_
best_model=grid_search.best_estimator_

model = DecisionTreeClassifier(random_state=42,
                              max_depth=2,
                              min_samples_split=22)
model.fit(df_X,y)

from sklearn import tree
tree.plot_tree(model)


from sklearn.metrics import confusion_matrix
# 아델리: 'A'
# 친스트랩: 'C'
y_true = np.array(['A', 'A', 'C', 'A', 'C', 'C', 'C'])
y_pred = np.array(['A', 'C', 'A', 'A', 'A', 'C', 'C'])

confusion_matrix(y_true=y_true, y_pred=y_pred)