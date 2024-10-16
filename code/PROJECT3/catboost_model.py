import pandas as pd
# 데이터 로드
df = pd.read_csv("../../bigfile/data_week3.csv")

# 범주형 변수를 One-Hot 인코딩 (get_dummies)
df_encoded = pd.get_dummies(df, columns=['unknown1'])

# 피처 선택 (인코딩된 피처 사용)
features = [col for col in df_encoded.columns if col != 'target']  # target을 제외한 모든 피처 사용
X = df_encoded[features]
y = df_encoded['target']

# 데이터셋을 훈련/검증 세트로 분할 (테스트 비율 5%, stratify 적용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# 훈련 세트에만 언더샘플링 적용
rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
# sampling_stategy = 1 -> 다수 클래스의 수가 소수 클래스 수의 100%로 줄어들음 (1일 때 제일 성능 ㄱㅊ)

# CatBoost
cat_model = CatBoostClassifier(iterations = 1500, depth= 10, learning_rate = 0.04, random_state=42, verbose=0)
cat_model.fit(X_train_resampled, y_train_resampled)
y_pred_cat = cat_model.predict(X_test)
recall_cat = recall_score(y_test, y_pred_cat)
print(f"Recall (CatBoost): {recall_cat}")