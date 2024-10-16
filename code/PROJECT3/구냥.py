## 예솔언니전처리 + 한렬님 lgbm
# Optimized Recall (LightGBM using Optuna): 0.6597510373443983
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
import optuna
from optuna.samplers import TPESampler

# 데이터 로드
df = pd.read_csv("../../bigfile/data_week3.csv")

# 한글 폰트 깨짐 방지
plt.rc('font', family='Malgun Gothic')

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

df.columns

X = df.drop(['unknown17','target'],axis=1)
y = df['target']
X.columns

# 데이터셋을 훈련/검증 세트로 분할 (테스트 비율 20%, stratify 적용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 훈련 세트에만 언더샘플링 적용
rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

# Optuna 최적화 함수 정의
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 200),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,  # random_state 고정
        'use_label_encoder': False  # 경고 방지
    }

    model = LGBMClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled)
    preds = model.predict(X_test)
    recall = recall_score(y_test, preds)

    return recall

# Optuna 스터디 설정 및 최적 파라미터 탐색 (시드 고정)
sampler = TPESampler(seed=42)  # Optuna 난수 시드 설정
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=50)

# 최적 하이퍼파라미터 출력
print("Best parameters (Optuna):", study.best_params)
print("Best recall (Optuna):", study.best_value)

# 최적의 파라미터로 LightGBM 모델 학습 및 예측
best_params = study.best_params
best_params['random_state'] = 42  # random_state 고정
lgb_model_optuna_final = LGBMClassifier(**best_params)
lgb_model_optuna_final.fit(X_train_resampled, y_train_resampled)

# 최종 모델로 예측 및 성능 평가
y_pred_optuna_final = lgb_model_optuna_final.predict(X_test)
recall_optuna_final = recall_score(y_test, y_pred_optuna_final)
print(f"Optimized Recall (LightGBM using Optuna): {recall_optuna_final}")

# ----------------------------------------------------------
## 한렬님 lgbm
# Optimized Recall (LightGBM using Optuna): 0.7510373443983402
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
import optuna
from optuna.samplers import TPESampler

# 데이터 로드
df = pd.read_csv("../../bigfile/data_week3.csv")

# 범주형 변수를 One-Hot 인코딩 (get_dummies)
df_encoded = pd.get_dummies(df, columns=['unknown1'])

# 피처 선택 (인코딩된 피처 사용)
features = [col for col in df_encoded.columns if col != 'target']  # target을 제외한 모든 피처 사용
X = df_encoded[features]
y = df_encoded['target']

# 데이터셋을 훈련/검증 세트로 분할 (테스트 비율 20%, stratify 적용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 훈련 세트에만 언더샘플링 적용
rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

# Optuna 최적화 함수 정의
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 200),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,  # random_state 고정
        'use_label_encoder': False  # 경고 방지
    }

    model = LGBMClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled)
    preds = model.predict(X_test)
    recall = recall_score(y_test, preds)

    return recall

# Optuna 스터디 설정 및 최적 파라미터 탐색 (시드 고정)
sampler = TPESampler(seed=42)  # Optuna 난수 시드 설정
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=50)

# 최적 하이퍼파라미터 출력
print("Best parameters (Optuna):", study.best_params)
print("Best recall (Optuna):", study.best_value)

# 최적의 파라미터로 LightGBM 모델 학습 및 예측
best_params = study.best_params
best_params['random_state'] = 42  # random_state 고정
lgb_model_optuna_final = LGBMClassifier(**best_params)
lgb_model_optuna_final.fit(X_train_resampled, y_train_resampled)

# 최종 모델로 예측 및 성능 평가
y_pred_optuna_final = lgb_model_optuna_final.predict(X_test)
recall_optuna_final = recall_score(y_test, y_pred_optuna_final)
print(f"Optimized Recall (LightGBM using Optuna): {recall_optuna_final}")
# -------------------------------------------------------------------
## 한렬님 lgbm code 이용한 xgboost
# Optimized Recall (XGBoost using Optuna): 0.6639004149377593
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
import optuna
from optuna.samplers import TPESampler

# 데이터 로드
df = pd.read_csv("../../bigfile/data_week3.csv")

# 범주형 변수를 One-Hot 인코딩 (get_dummies)
df_encoded = pd.get_dummies(df, columns=['unknown1'])

# 피처 선택 (인코딩된 피처 사용)
features = [col for col in df_encoded.columns if col != 'target']  # target을 제외한 모든 피처 사용
X = df_encoded[features]
y = df_encoded['target']

# 데이터셋을 훈련/검증 세트로 분할 (테스트 비율 20%, stratify 적용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 훈련 세트에만 언더샘플링 적용
rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

# Optuna 최적화 함수 정의
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 200),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,  # random_state 고정
        'use_label_encoder': False,  # 경고 방지
        'eval_metric': 'logloss'  # 평가 지표 설정
    }

    model = XGBClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled)
    preds = model.predict(X_test)
    recall = recall_score(y_test, preds)

    return recall

# Optuna 스터디 설정 및 최적 파라미터 탐색 (시드 고정)
sampler = TPESampler(seed=42)  # Optuna 난수 시드 설정
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=50)

# 최적 하이퍼파라미터 출력
print("Best parameters (Optuna):", study.best_params)
print("Best recall (Optuna):", study.best_value)

# 최적의 파라미터로 XGBoost 모델 학습 및 예측
best_params = study.best_params
best_params['random_state'] = 42  # random_state 고정
xgb_model_optuna_final = XGBClassifier(**best_params)
xgb_model_optuna_final.fit(X_train_resampled, y_train_resampled)

# 최종 모델로 예측 및 성능 평가
y_pred_optuna_final = xgb_model_optuna_final.predict(X_test)
recall_optuna_final = recall_score(y_test, y_pred_optuna_final)
print(f"Optimized Recall (XGBoost using Optuna): {recall_optuna_final}")
# ------------------------------------------------------------------
## 예솔언니 전처리 + 한렬님 lgbm code 이용한 xgboost
## 예솔언니전처리 + 한렬님 lgbm
# Optimized Recall (XGBoost using Optuna): 0.6721991701244814
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
import optuna
from optuna.samplers import TPESampler

# 데이터 로드
df = pd.read_csv("../../bigfile/data_week3.csv")

# 한글 폰트 깨짐 방지
plt.rc('font', family='Malgun Gothic')

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

df.columns

X = df.drop(['unknown17','target'],axis=1)
y = df['target']
X.columns

# 데이터셋을 훈련/검증 세트로 분할 (테스트 비율 20%, stratify 적용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 훈련 세트에만 언더샘플링 적용
rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

# Optuna 최적화 함수 정의
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 200),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,  # random_state 고정
        'use_label_encoder': False,  # 경고 방지
        'eval_metric': 'logloss'  # 평가 지표 설정
    }

    model = XGBClassifier(**params)
    model.fit(X_train_resampled, y_train_resampled)
    preds = model.predict(X_test)
    recall = recall_score(y_test, preds)

    return recall

# Optuna 스터디 설정 및 최적 파라미터 탐색 (시드 고정)
sampler = TPESampler(seed=42)  # Optuna 난수 시드 설정
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=50)

# 최적 하이퍼파라미터 출력
print("Best parameters (Optuna):", study.best_params)
print("Best recall (Optuna):", study.best_value)

# 최적의 파라미터로 XGBoost 모델 학습 및 예측
best_params = study.best_params
best_params['random_state'] = 42  # random_state 고정
xgb_model_optuna_final = XGBClassifier(**best_params)
xgb_model_optuna_final.fit(X_train_resampled, y_train_resampled)

# 최종 모델로 예측 및 성능 평가
y_pred_optuna_final = xgb_model_optuna_final.predict(X_test)
recall_optuna_final = recall_score(y_test, y_pred_optuna_final)
print(f"Optimized Recall (XGBoost using Optuna): {recall_optuna_final}")