import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the CSV file to inspect the contents
raw_data = pd.read_csv('../../bigfile/1주_실습데이터.csv')
data = raw_data.copy()

# 원본 데이터 나누기
X = data.drop(columns=['Y'])
y = data['Y']

# 의사결정나무 모델 학습 (최대 깊이를 제한해서 트리가 너무 복잡해지지 않도록 함)
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X, y)

# 의사결정나무 시각화 (큰 이미지 크기로 설정)
plt.figure(figsize=(15, 10))  # 이미지 크기를 크게 설정
plot_tree(tree_model, feature_names=X.columns, class_names=['Class 0', 'Class 1'], filled=True, rounded=True)
plt.title("Decision Tree Visualization with max_depth=3")
plt.show()

# X3이 0.406 이하일 때의 Y 값 분포 확인
below_threshold = data[data['X3'] <= 0.406]['Y'].value_counts()

# X3이 0.406보다 클 때의 Y 값 분포 확인
above_threshold = data[data['X3'] > 0.406]['Y'].value_counts()

# 결과 출력
print("X3 <= 0.406일 때 Y 값 분포:")
print(below_threshold)

print("\nX3 > 0.406일 때 Y 값 분포:")
print(above_threshold)

# ---------------------------------------------------
# 원본 데이터 나누기
X = data.drop(columns=['Y'])
y = data['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

data.shape
X.shape
X_train

# 의사결정나무 모델 학습 (최대 깊이를 제한해서 트리가 너무 복잡해지지 않도록 함)
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

# 의사결정나무 시각화 (큰 이미지 크기로 설정)
plt.figure(figsize=(15, 10))  # 이미지 크기를 크게 설정
plot_tree(tree_model, feature_names=X.columns, class_names=['Class 0', 'Class 1'], filled=True, rounded=True)
plt.title("Decision Tree Visualization with max_depth=3")
plt.show()

# X3이 0.406 이하일 때의 Y 값 분포 확인
below_threshold = data[data['X3'] <= 0.406]['Y'].value_counts()

# X3이 0.406보다 클 때의 Y 값 분포 확인
above_threshold = data[data['X3'] > 0.406]['Y'].value_counts()

# 결과 출력
print("X3 <= 0.406일 때 Y 값 분포:")
print(below_threshold)

print("\nX3 > 0.406일 때 Y 값 분포:")
print(above_threshold)