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

# 범주형 변수를 One-Hot 인코딩 (get_dummies)
df_encoded = pd.get_dummies(df, columns=['unknown1'])
df_encoded.info()
# ---------------------------------------------------

# Re-run KMeans clustering on 'unknown17'
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
df_encoded['unknown17_cluster'] = kmeans.fit_predict(df_encoded[['unknown17']])
df_encoded['unknown17_cluster'] = df_encoded['unknown17_cluster'].astype('object')

# Perform one-hot encoding on 'unknown17_cluster'
df_encoded_2 = pd.get_dummies(df_encoded, columns=['unknown17_cluster'], drop_first=True)
# ---------------------------------------------------

# X,y 나누기
X = df_encoded_2.drop(['target', 'unknown17'], axis=1)
X.info()
y = df_encoded['target']

# 데이터셋을 훈련/검증 세트로 분할 (테스트 비율 10%, stratify 적용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
print(f"Before Sampling: {Counter(y_train)}")
# {0: 10313, 1: 962}

# 모델 및 샘플링 하기
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


### 1. 토멕링크
# Tomek Links
from imblearn.under_sampling import TomekLinks
tl = TomekLinks(sampling_strategy='majority')
X_train_TomekLinks, y_train_TomekLinks = tl.fit_resample(X_train,y_train)

print(f"Before Sampling: {Counter(y_train)}")
print(f"After Sampling, Tomek Links: {Counter(y_train_TomekLinks)}")
# {0: 9911, 1: 962}


## Logistic Regression (로지스틱 회귀 분석)
model_Tomek_LogisticRegression = LogisticRegression()
model_Tomek_LogisticRegression.fit(X_train_TomekLinks, y_train_TomekLinks)

y_pred_Tomek_LogisticRegression = model_Tomek_LogisticRegression.predict(X_test)

## Decision Tree Regressor (의사결정나무 분류)
model_Tomek_DecisionTreeClassifier = DecisionTreeClassifier(random_state=42)
model_Tomek_DecisionTreeClassifier.fit(X_train_TomekLinks, y_train_TomekLinks)

y_pred_Tomek_DecisionTreeClassifier = model_Tomek_DecisionTreeClassifier.predict(X_test)

## Random Forest (랜덤포레스트)
model_Tomek_RandomForest = RandomForestClassifier(n_estimators=100, random_state=42)
model_Tomek_RandomForest.fit(X_train_TomekLinks, y_train_TomekLinks)

y_pred_Tomek_RandomForest = model_Tomek_RandomForest.predict(X_test)

## XG Boost
model_Tomek_XGBoost = XGBClassifier(n_estimators=100, random_state=42)
model_Tomek_XGBoost.fit(X_train_TomekLinks, y_train_TomekLinks)

y_pred_Tomek_XGBoost = model_Tomek_XGBoost.predict(X_test)

### Light GBM
model_Tomek_LightGBM = LGBMClassifier(n_estimators=100, random_state=42, verbose=0)
model_Tomek_LightGBM.fit(X_train_TomekLinks, y_train_TomekLinks)

y_pred_Tomek_LightGBM = model_Tomek_LightGBM.predict(X_test)

### Cat Boost
model_Tomek_CatBoost = CatBoostClassifier(n_estimators=100, random_state=42)
model_Tomek_CatBoost.fit(X_train_TomekLinks, y_train_TomekLinks)

y_pred_Tomek_CatBoost = model_Tomek_CatBoost.predict(X_test)



### 2. 스모트
# SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)

print(f"Before Sampling: {Counter(y_train)}")
print(f"After Sampling, SMOTE: {Counter(y_train_SMOTE)}")
# {0: 10313, 1: 10313}

## Logistic Regression (로지스틱 회귀 분석)
model_SMOTE_LogisticRegression = LogisticRegression()
model_SMOTE_LogisticRegression.fit(X_train_SMOTE, y_train_SMOTE)

y_pred_SMOTE_LogisticRegression = model_SMOTE_LogisticRegression.predict(X_test)

## Decision Tree Regressor (의사결정나무 분류)
model_SMOTE_DecisionTreeClassifier = DecisionTreeClassifier(random_state=42)
model_SMOTE_DecisionTreeClassifier.fit(X_train_SMOTE, y_train_SMOTE)

y_pred_SMOTE_DecisionTreeClassifier = model_SMOTE_DecisionTreeClassifier.predict(X_test)

## Random Forest (랜덤포레스트)
model_SMOTE_RandomForest = RandomForestClassifier(n_estimators=100, random_state=42)
model_SMOTE_RandomForest.fit(X_train_SMOTE, y_train_SMOTE)

y_pred_SMOTE_RandomForest = model_SMOTE_RandomForest.predict(X_test)

## XG Boost
model_SMOTE_XGBoost = XGBClassifier(n_estimators=100, random_state=42)
model_SMOTE_XGBoost.fit(X_train_SMOTE, y_train_SMOTE)

y_pred_SMOTE_XGBoost = model_SMOTE_XGBoost.predict(X_test)

### Light GBM
model_SMOTE_LightGBM = LGBMClassifier(n_estimators=100, random_state=42, verbose=0)
model_SMOTE_LightGBM.fit(X_train_SMOTE, y_train_SMOTE)

y_pred_SMOTE_LightGBM = model_SMOTE_LightGBM.predict(X_test)

### Cat Boost
model_SMOTE_CatBoost = CatBoostClassifier(n_estimators=100, random_state=42)
model_SMOTE_CatBoost.fit(X_train_SMOTE, y_train_SMOTE)

y_pred_SMOTE_CatBoost = model_SMOTE_CatBoost.predict(X_test)



### 3. 스모트 토멕
# SMOTE TOMEK
from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_train_SmoteTomek, y_train_SmoteTomek = smote_tomek.fit_resample(X_train, y_train)

print(f"Before Sampling: {Counter(y_train)}")
print(f"After Sampling, SMOTE TOMEK: {Counter(y_train_SmoteTomek)}")
# {0: 10020, 1: 10020}


## Logistic Regression (로지스틱 회귀 분석)
model_SmoteTomek_LogisticRegression = LogisticRegression()
model_SmoteTomek_LogisticRegression.fit(X_train_SmoteTomek, y_train_SmoteTomek)

y_pred_SmoteTomek_LogisticRegression = model_SmoteTomek_LogisticRegression.predict(X_test)

## Decision Tree Regressor (의사결정나무 분류)
model_SmoteTomek_DecisionTreeClassifier = DecisionTreeClassifier(random_state=42)
model_SmoteTomek_DecisionTreeClassifier.fit(X_train_SmoteTomek, y_train_SmoteTomek)

y_pred_SmoteTomek_DecisionTreeClassifier = model_SmoteTomek_DecisionTreeClassifier.predict(X_test)

## Random Forest (랜덤포레스트)
model_SmoteTomek_RandomForest = RandomForestClassifier(n_estimators=100, random_state=42)
model_SmoteTomek_RandomForest.fit(X_train_SmoteTomek, y_train_SmoteTomek)

y_pred_SmoteTomek_RandomForest = model_SmoteTomek_RandomForest.predict(X_test)

## XG Boost
model_SmoteTomek_XGBoost = XGBClassifier(n_estimators=100, random_state=42)
model_SmoteTomek_XGBoost.fit(X_train_SmoteTomek, y_train_SmoteTomek)

y_pred_SmoteTomek_XGBoost = model_SmoteTomek_XGBoost.predict(X_test)

### Light GBM
model_SmoteTomek_LightGBM = LGBMClassifier(n_estimators=100, random_state=42, verbose=0)
model_SmoteTomek_LightGBM.fit(X_train_SmoteTomek, y_train_SmoteTomek)

y_pred_SmoteTomek_LightGBM = model_SmoteTomek_LightGBM.predict(X_test)

### Cat Boost
model_SmoteTomek_CatBoost = CatBoostClassifier(n_estimators=100, random_state=42)
model_SmoteTomek_CatBoost.fit(X_train_SmoteTomek, y_train_SmoteTomek)

y_pred_SmoteTomek_CatBoost = model_SmoteTomek_CatBoost.predict(X_test)


### 4. Tomek Links
from imblearn.under_sampling import TomekLinks
tl = TomekLinks(sampling_strategy='majority')
X_train_Tomek, y_train_Tomek = tl.fit_resample(X_train, y_train)

print(f"Before Sampling: {Counter(y_train)}")
print(f"After Sampling, Tomek Links: {Counter(y_train_Tomek)}")
# {0: 9911, 1: 962}

## Logistic Regression (로지스틱 회귀 분석)
model_Tomek_LogisticRegression = LogisticRegression()
model_Tomek_LogisticRegression.fit(X_train_Tomek, y_train_Tomek)

y_pred_Tomek_LogisticRegression = model_Tomek_LogisticRegression.predict(X_test)

## Decision Tree Regressor (의사결정나무 분류)
model_Tomek_DecisionTreeClassifier = DecisionTreeClassifier(random_state=42)
model_Tomek_DecisionTreeClassifier.fit(X_train_Tomek, y_train_Tomek)

y_pred_Tomek_DecisionTreeClassifier = model_Tomek_DecisionTreeClassifier.predict(X_test)

## Random Forest (랜덤포레스트)
model_Tomek_RandomForest = RandomForestClassifier(n_estimators=100, random_state=42)
model_Tomek_RandomForest.fit(X_train_Tomek, y_train_Tomek)

y_pred_Tomek_RandomForest = model_Tomek_RandomForest.predict(X_test)

## XG Boost
model_Tomek_XGBoost = XGBClassifier(n_estimators=100, random_state=42)
model_Tomek_XGBoost.fit(X_train_Tomek, y_train_Tomek)

y_pred_Tomek_XGBoost = model_Tomek_XGBoost.predict(X_test)

### Light GBM
model_Tomek_LightGBM = LGBMClassifier(n_estimators=100, random_state=42, verbose=0)
model_Tomek_LightGBM.fit(X_train_Tomek, y_train_Tomek)

y_pred_Tomek_LightGBM = model_Tomek_LightGBM.predict(X_test)

### Cat Boost
model_Tomek_CatBoost = CatBoostClassifier(n_estimators=100, random_state=42)
model_Tomek_CatBoost.fit(X_train_Tomek, y_train_Tomek)

y_pred_Tomek_CatBoost = model_Tomek_CatBoost.predict(X_test)


### 5. RandomOverSampler
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

print(f"Before Sampling: {Counter(y_train)}")
print(f"After Sampling, Random Over Sampling: {Counter(y_train_ros)}")
# {0: 10313, 1: 10313}

## Logistic Regression (로지스틱 회귀 분석)
model_ROS_LogisticRegression = LogisticRegression()
model_ROS_LogisticRegression.fit(X_train_ros, y_train_ros)

y_pred_ROS_LogisticRegression = model_ROS_LogisticRegression.predict(X_test)

## Decision Tree Regressor (의사결정나무 분류)
model_ROS_DecisionTreeClassifier = DecisionTreeClassifier(random_state=42)
model_ROS_DecisionTreeClassifier.fit(X_train_ros, y_train_ros)

y_pred_ROS_DecisionTreeClassifier = model_ROS_DecisionTreeClassifier.predict(X_test)

## Random Forest (랜덤포레스트)
model_ROS_RandomForest = RandomForestClassifier(n_estimators=100, random_state=42)
model_ROS_RandomForest.fit(X_train_ros, y_train_ros)

y_pred_ROS_RandomForest = model_ROS_RandomForest.predict(X_test)

## XG Boost
model_ROS_XGBoost = XGBClassifier(n_estimators=100, random_state=42)
model_ROS_XGBoost.fit(X_train_ros, y_train_ros)

y_pred_ROS_XGBoost = model_ROS_XGBoost.predict(X_test)

### Light GBM
model_ROS_LightGBM = LGBMClassifier(n_estimators=100, random_state=42, verbose=0)
model_ROS_LightGBM.fit(X_train_ros, y_train_ros)

y_pred_ROS_LightGBM = model_ROS_LightGBM.predict(X_test)

### Cat Boost
model_ROS_CatBoost = CatBoostClassifier(n_estimators=100, random_state=42)
model_ROS_CatBoost.fit(X_train_ros, y_train_ros)

y_pred_ROS_CatBoost = model_ROS_CatBoost.predict(X_test)


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
from sklearn.metrics import recall_score

recall_SMOTE_LogisticRegression = recall_score(y_test, y_pred_SMOTE_LogisticRegression) # 0.60
recall_SMOTE_DecisionTreeClassifier = recall_score(y_test, y_pred_SMOTE_DecisionTreeClassifier)
recall_SMOTE_RandomForest = recall_score(y_test, y_pred_SMOTE_RandomForest)
recall_SMOTE_XGBoost = recall_score(y_test, y_pred_SMOTE_XGBoost)
recall_SMOTE_LightGBM = recall_score(y_test, y_pred_SMOTE_LightGBM)
recall_SMOTE_CatBoost = recall_score(y_test, y_pred_SMOTE_CatBoost)

recall_SmoteTomek_LogisticRegression = recall_score(y_test, y_pred_SmoteTomek_LogisticRegression)
recall_SmoteTomek_DecisionTreeClassifier = recall_score(y_test, y_pred_SmoteTomek_DecisionTreeClassifier)
recall_SmoteTomek_RandomForest = recall_score(y_test, y_pred_SmoteTomek_RandomForest)
recall_SmoteTomek_XGBoost = recall_score(y_test, y_pred_SmoteTomek_XGBoost)
recall_SmoteTomek_LightGBM = recall_score(y_test, y_pred_SmoteTomek_LightGBM)
recall_SmoteTomek_CatBoost = recall_score(y_test, y_pred_SmoteTomek_CatBoost)

recall_Tomek_LogisticRegression = recall_score(y_test, y_pred_Tomek_LogisticRegression)
recall_Tomek_DecisionTreeClassifier = recall_score(y_test, y_pred_Tomek_DecisionTreeClassifier)
recall_Tomek_RandomForest = recall_score(y_test, y_pred_Tomek_RandomForest)
recall_Tomek_XGBoost = recall_score(y_test, y_pred_Tomek_XGBoost)
recall_Tomek_LightGBM = recall_score(y_test, y_pred_Tomek_LightGBM)
recall_Tomek_CatBoost = recall_score(y_test, y_pred_Tomek_CatBoost)

recall_ROS_LogisticRegression = recall_score(y_test, y_pred_ROS_LogisticRegression)
recall_ROS_DecisionTreeClassifier = recall_score(y_test, y_pred_ROS_DecisionTreeClassifier)
recall_ROS_RandomForest = recall_score(y_test, y_pred_ROS_RandomForest)
recall_ROS_XGBoost = recall_score(y_test, y_pred_ROS_XGBoost)
recall_ROS_LightGBM = recall_score(y_test, y_pred_ROS_LightGBM)
recall_ROS_CatBoost = recall_score(y_test, y_pred_ROS_CatBoost)

### recall_score
recall_RUS_LogisticRegression = recall_score(y_test, y_pred_RUS_LogisticRegression)
recall_RUS_DecisionTreeClassifier = recall_score(y_test, y_pred_RUS_DecisionTreeClassifier)
recall_RUS_RandomForest = recall_score(y_test, y_pred_RUS_RandomForest) # 0.6166666666666667
recall_RUS_XGBoost = recall_score(y_test, y_pred_RUS_XGBoost)
recall_RUS_LightGBM = recall_score(y_test, y_pred_RUS_LightGBM) # 0.6307053941908713
recall_RUS_CatBoost = recall_score(y_test, y_pred_RUS_CatBoost) # test_size 0.1, 0.6583
















coefficients = model_SMOTE_LogisticRegression.coef_[0]
model_SMOTE_LogisticRegression.intercept_[0]
# 특성 이름과 함께 계수를 출력
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False)
feature_importance