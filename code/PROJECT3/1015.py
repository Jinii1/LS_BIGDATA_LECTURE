from xgboost import XGBClassifier
from sklearn.metrics import recall_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import pandas as pd

# 데이터 로드
df = pd.read_csv("../../bigfile/data_week3.csv")

# 범주형 변수를 One-Hot 인코딩 (get_dummies)
df_encoded = pd.get_dummies(df, columns=['unknown1'])
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


# 피처 선택 (인코딩된 피처 사용)
features = [col for col in df_encoded.columns if col != 'target']  # target을 제외한 모든 피처 사용
X = df_encoded[features]
y = df_encoded['target']

# 데이터셋을 훈련/검증 세트로 분할 (테스트 비율 20%, stratify 적용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 훈련 세트에만 언더샘플링 적용
rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

# XGBClassifier 적용
xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_resampled, y_train_resampled)

# 테스트 데이터에 대한 예측
y_pred_xgb = xgb_model.predict(X_test)

# Recall 계산 및 결과 출력
recall_xgb = recall_score(y_test, y_pred_xgb)
print(f"Recall (XGBoost): {recall_xgb}")

# --
from sklearn.model_selection import RandomizedSearchCV

# 하이퍼파라미터 튜닝을 위한 파라미터 그리드 설정
param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'min_child_weight': [1, 3, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
}

# RandomizedSearchCV 적용
random_search = RandomizedSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                                   param_distributions,
                                   n_iter=50,
                                   scoring='recall',
                                   cv=5,
                                   random_state=42)

# 모델 훈련
random_search.fit(X_train_resampled, y_train_resampled)

# 최적 하이퍼파라미터 출력
print("Best parameters found by Random Search:", random_search.best_params_)

# 최적의 모델 사용
best_xgb_model = random_search.best_estimator_

# 테스트 데이터에 대한 예측
y_pred_best_xgb = best_xgb_model.predict(X_test)

# Recall 계산 및 결과 출력
recall_best_xgb = recall_score(y_test, y_pred_best_xgb)
print(f"Recall (Best XGBoost): {recall_best_xgb}")

# --
# 최적의 하이퍼파라미터로 모델 재훈련
final_xgb_model = XGBClassifier(
    subsample=0.5,
    n_estimators=100,
    min_child_weight=5,
    max_depth=None,
    learning_rate=0.1,
    colsample_bytree=1.0,
    eval_metric='logloss',
    use_label_encoder=False
)

final_xgb_model.fit(X_train_resampled, y_train_resampled)

# 테스트 데이터에 대한 예측
y_pred_final_xgb = final_xgb_model.predict(X_test)

# Recall 계산 및 결과 출력
final_recall = recall_score(y_test, y_pred_final_xgb)
print(f"Final Recall (XGBoost): {final_recall}")

# --
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBClassifier

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'eval_metric': 'logloss',
        'use_label_encoder': False
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

# Optuna를 사용한 하이퍼파라미터 최적화
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best parameters:", study.best_params)
print("Best recall:", study.best_value)

# --
# 최적의 하이퍼파라미터로 모델 재학습
best_params = study.best_params  # Optuna에서 찾은 최적의 파라미터
best_params['random_state'] = 42  # random_state 추가
best_recall = study.best_value   # 최적의 리콜 값

# XGBClassifier 초기화
xgb_best_model = XGBClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    min_child_weight=best_params['min_child_weight'],
    use_label_encoder=False,
    eval_metric='logloss'  # 또는 다른 평가지표
)

# 모델 훈련
xgb_best_model.fit(X_train_resampled, y_train_resampled)

# 테스트 데이터에 대한 예측
y_pred_best = xgb_best_model.predict(X_test)


# 최종 리콜 계산
final_recall = recall_score(y_test, y_pred_best)
print(f"Final Recall (Best XGBoost): {final_recall}") #0.6887966804979253