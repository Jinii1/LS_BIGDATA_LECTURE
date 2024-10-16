import pandas as pd
import numpy as np
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.under_sampling import RandomUnderSampler
import optuna
from lightgbm import LGBMClassifier

# 데이터 로드 및 전처리
df = pd.read_csv("../../bigfile/data_week3.csv")

# 범주형 변수를 One-Hot 인코딩 (get_dummies)
df['unknown2'].value_counts().sort_index(ascending=False) # 294451, 1
df = df[df['unknown2'] != 294451]

df['unknown5'].value_counts().sort_index(ascending=False) # 27, 1
df = df[df['unknown5'] != 27]

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

# --------------------------------------------------
# RandomizedSearchCV를 사용한 LightGBM 하이퍼파라미터 튜닝
param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, -1],
    'num_leaves': [31, 64, 128],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'min_child_weight': [1e-3, 1e-2, 1e-1],
}

random_search = RandomizedSearchCV(LGBMClassifier(random_state=42),
                                   param_distributions,
                                   n_iter=50,
                                   scoring='recall',
                                   cv=5,
                                   random_state=42)

random_search.fit(X_train_resampled, y_train_resampled)

print("Best parameters found by Random Search:", random_search.best_params_)

# 최적의 모델로 예측
best_lgb_model = random_search.best_estimator_
y_pred_best_lgb = best_lgb_model.predict(X_test)
recall_best_lgb = recall_score(y_test, y_pred_best_lgb)
print(f"Recall (Best LightGBM): {recall_best_lgb}")

# --------------------------------------------------
# Optuna를 사용한 하이퍼파라미터 최적화
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'num_leaves': trial.suggest_int('num_leaves', 31, 128),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 1e-1),
    }

    model = LGBMClassifier(**params, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    
    return recall

# Optuna 최적화 실행
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best parameters found by Optuna:", study.best_params)
print("Best recall achieved by Optuna:", study.best_value)

# 최적의 하이퍼파라미터로 LightGBM 모델 학습
best_params = study.best_params
lgb_best_model_optuna = LGBMClassifier(**best_params, random_state=42)
lgb_best_model_optuna.fit(X_train_resampled, y_train_resampled)

# 테스트 데이터에 대해 예측 및 평가
y_pred_best_optuna = lgb_best_model_optuna.predict(X_test)
recall_best_optuna = recall_score(y_test, y_pred_best_optuna)
print(f"Final Recall (Optuna Best LightGBM): {recall_best_optuna}") # 0.6473029045643154