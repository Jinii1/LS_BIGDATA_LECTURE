import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from catboost import CatBoostClassifier

# 한글 폰트 깨짐 방지
plt.rc('font', family='Malgun Gothic')

df = pd.read_csv("../../bigfile/data_week3.csv")
df.info()
df = pd.get_dummies(df, columns=['unknown1'], drop_first=True)
df['unknown1_type2'] = df['unknown1_type2'].astype('int64')
df['unknown1_type3'] = df['unknown1_type3'].astype('int64')
df['unknown1_type4'] = df['unknown1_type4'].astype('int64')

df.info()

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

X = df.drop(['unknown17','target'],axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


### 0. 기본 데이터(샘플링 안 함)
# Original Data
X_train
y_train


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
## Logistic Regression (로지스틱 회귀 분석)
model_Original_LogisticRegression = LogisticRegression()
model_Original_LogisticRegression.fit(X_train, y_train)

y_pred_Original_LogisticRegression = model_Original_LogisticRegression.predict(X_test)

## Decision Tree Regressor (의사결정나무 분류)
model_Original_DecisionTreeClassifier = DecisionTreeClassifier(random_state=42)
model_Original_DecisionTreeClassifier.fit(X_train, y_train)

y_pred_Original_DecisionTreeClassifier = model_Original_DecisionTreeClassifier.predict(X_test)

## Random Forest (랜덤포레스트)
model_Original_RandomForest = RandomForestClassifier(n_estimators=100, random_state=42)
model_Original_RandomForest.fit(X_train, y_train)

y_pred_Original_RandomForest = model_Original_RandomForest.predict(X_test)

## XG Boost
model_Original_XGBoost = XGBClassifier(n_estimators=100, random_state=42)
model_Original_XGBoost.fit(X_train, y_train)

y_pred_Original_XGBoost = model_Original_XGBoost.predict(X_test)

### Light GBM
model_Original_LightGBM = LGBMClassifier(n_estimators=100, random_state=42, verbose=0) # verbose=0 하면 셀 실행시 결과 보여주는 창에 글자들 없어짐.
model_Original_LightGBM.fit(X_train, y_train)

y_pred_Original_LightGBM = model_Original_LightGBM.predict(X_test)

### Cat Boost
model_Original_CatBoost = CatBoostClassifier(n_estimators=100, random_state=42)
model_Original_CatBoost.fit(X_train, y_train)

y_pred_Original_CatBoost = model_Original_CatBoost.predict(X_test)

from sklearn.metrics import recall_score

recall_Original_LogisticRegression = recall_score(y_test, y_pred_Original_LogisticRegression)
recall_Original_DecisionTreeClassifier = recall_score(y_test, y_pred_Original_DecisionTreeClassifier)
recall_Original_RandomForest = recall_score(y_test, y_pred_Original_RandomForest)
recall_Original_XGBoost = recall_score(y_test, y_pred_Original_XGBoost)
recall_Original_LightGBM = recall_score(y_test, y_pred_Original_LightGBM)
recall_Original_CatBoost = recall_score(y_test, y_pred_Original_CatBoost)










































# 샘플링 기법 정의
sampler_under = RandomUnderSampler(random_state=42)
sampler_over = RandomOverSampler(random_state=42)
sampler_smote = SMOTE(random_state=42)

X_train_resampled_under, y_train_resampled_under = sampler_under.fit_resample(X_train, y_train)
X_train_resampled_over, y_train_resampled_over = sampler_over.fit_resample(X_train, y_train)
X_train_resampled_smote, y_train_resampled_smote = sampler_smote.fit_resample(X_train, y_train)

model_CatBoost_origin = CatBoostClassifier(random_state=42)
model_CatBoost_under = CatBoostClassifier(random_state=42)
model_CatBoost_over = CatBoostClassifier(random_state=42)
model_CatBoost_smote = CatBoostClassifier(random_state=42)

# iterations=1000, learning_rate=0.03, depth=6, 

# 모델 학습
model_CatBoost_origin.fit(X_train, y_train)
model_CatBoost_under.fit(X_train_resampled_under, y_train_resampled_under)
model_CatBoost_over.fit(X_train_resampled_over, y_train_resampled_over)
model_CatBoost_smote.fit(X_train_resampled_smote, y_train_resampled_smote)

# 예측 및 재현율 계산
y_pred_CatBoost_origin = model_CatBoost_origin.predict(X_test)
y_pred_CatBoost_under = model_CatBoost_under.predict(X_test)
y_pred_CatBoost_over = model_CatBoost_over.predict(X_test)
y_pred_CatBoost_smote = model_CatBoost_smote.predict(X_test)

recall_CatBoost_origin = recall_score(y_test, y_pred_CatBoost_origin)
recall_CatBoost_under = recall_score(y_test, y_pred_CatBoost_under)
recall_CatBoost_over = recall_score(y_test, y_pred_CatBoost_over)
recall_CatBoost_smote = recall_score(y_test, y_pred_CatBoost_smote)

# 결과 출력
print(f"재현율(Recall_Catboost_origin): {recall_CatBoost_origin:.5f}")
print(f"재현율(Recall_Catboost_under): {recall_CatBoost_under:.5f}")
print(f"재현율(Recall_Catboost_over): {recall_CatBoost_over:.5f}")
print(f"재현율(Recall_Catboost_smote): {recall_CatBoost_smote:.5f}")

############# 그냥

# 재현율(Recall_Catboost_origin): 0.02490
# 재현율(Recall_Catboost_under): 0.63071
# 재현율(Recall_Catboost_over): 0.25726
# 재현율(Recall_Catboost_smote): 0.28216

################## X17제거

# 재현율(Recall_Catboost_origin): 0.01660
# 재현율(Recall_Catboost_under): 0.61411
# 재현율(Recall_Catboost_over): 0.22822
# 재현율(Recall_Catboost_smote): 0.26141

##################### 

# 재현율(Recall_Catboost_origin): 0.01660
# 재현율(Recall_Catboost_under): 0.70124
# 재현율(Recall_Catboost_over): 0.27386
# 재현율(Recall_Catboost_smote): 0.29876