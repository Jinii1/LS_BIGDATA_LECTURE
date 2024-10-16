import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from imblearn.under_sampling import TomekLinks
from collections import Counter # 샘플링 갯수 결과 보기 위해 필요한 모듈

# 데이터 로드
df = pd.read_csv("../../bigfile/data_week3.csv")

df['unknown2'].value_counts().sort_index(ascending=False) # 294451, 1
df = df[df['unknown2'] != 294451]

df['unknown5'].value_counts().sort_index(ascending=False) # 27, 1
df = df[df['unknown2'] != 294451]

df['unknown6'].value_counts().sort_index(ascending=False) # 2398, 1
df = df[df['unknown6'] != 2398]

df['unknown8'].value_counts().sort_index(ascending=False) # 31706, 1
df = df[df['unknown8'] != 31706]

df['unknown10'].value_counts().sort_index(ascending=False) # 877, 1
df = df[df['unknown10'] != 877]

df['unknown14'].value_counts().sort_index(ascending=False) # 403, 1
df = df[df['unknown14'] != 403]

df['unknown16'].value_counts().sort_index(ascending=False) # 2840.5, 1
df = df[df['unknown16'] != 2840.5]

columns_to_drop = ['unknown4', 'unknown5', 'unknown9', 'unknown12', 'unknown14', 'unknown15', 'unknown17']
df = df.drop(columns_to_drop, axis=1)
df

# 범주형 변수를 One-Hot 인코딩 (get_dummies)
df_encoded = pd.get_dummies(df, columns=['unknown1'])

# X,y 나누기
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

# 데이터셋을 훈련/검증 세트로 분할 (테스트 비율 20%, stratify 적용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Before Sampling: {Counter(y_train)}")
# {0: 10313, 1: 962}

# 모델 및 샘플링 하기
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

### 6. random under sampling
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

print(f"Before Sampling: {Counter(y_train)}")
print(f"After Sampling, Random Under Sampling: {Counter(y_train_rus)}")
# {0: 962, 1: 962}

## Logistic Regression (로지스틱 회귀 분석)
model_RUS_LogisticRegression = LogisticRegression()
model_RUS_LogisticRegression.fit(X_train_rus, y_train_rus)

y_pred_RUS_LogisticRegression = model_RUS_LogisticRegression.predict(X_test)

## Decision Tree Regressor (의사결정나무 분류)
model_RUS_DecisionTreeClassifier = DecisionTreeClassifier(random_state=42)
model_RUS_DecisionTreeClassifier.fit(X_train_rus, y_train_rus)

y_pred_RUS_DecisionTreeClassifier = model_RUS_DecisionTreeClassifier.predict(X_test)

## Random Forest (랜덤포레스트)
model_RUS_RandomForest = RandomForestClassifier(n_estimators=100, random_state=42)
model_RUS_RandomForest.fit(X_train_rus, y_train_rus)

y_pred_RUS_RandomForest = model_RUS_RandomForest.predict(X_test)

## XG Boost
model_RUS_XGBoost = XGBClassifier(n_estimators=100, random_state=42)
model_RUS_XGBoost.fit(X_train_rus, y_train_rus)

y_pred_RUS_XGBoost = model_RUS_XGBoost.predict(X_test)

### Light GBM
model_RUS_LightGBM = LGBMClassifier(n_estimators=100, random_state=42, verbose=0)
model_RUS_LightGBM.fit(X_train_rus, y_train_rus)

y_pred_RUS_LightGBM = model_RUS_LightGBM.predict(X_test)

### Cat Boost
model_RUS_CatBoost = CatBoostClassifier(n_estimators=100, random_state=42)
model_RUS_CatBoost.fit(X_train_rus, y_train_rus)

y_pred_RUS_CatBoost = model_RUS_CatBoost.predict(X_test)

### recall_score
recall_RUS_LogisticRegression = recall_score(y_test, y_pred_RUS_LogisticRegression)
recall_RUS_DecisionTreeClassifier = recall_score(y_test, y_pred_RUS_DecisionTreeClassifier)
recall_RUS_RandomForest = recall_score(y_test, y_pred_RUS_RandomForest)
recall_RUS_XGBoost = recall_score(y_test, y_pred_RUS_XGBoost)
recall_RUS_LightGBM = recall_score(y_test, y_pred_RUS_LightGBM)
recall_RUS_CatBoost = recall_score(y_test, y_pred_RUS_CatBoost) # 0.6597510373443983

# x4 5 9 12 14 15 제거 해봐도 되고 안 해봐도 되고

from sklearn.metrics import confusion_matrix
# 혼동행렬 계산
conf_matrix = confusion_matrix(y_test, y_pred_RUS_LightGBM)

# 혼동행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for LightGBM')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()