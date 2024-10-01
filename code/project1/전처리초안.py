import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file to inspect the contents
raw_data = pd.read_csv('../../bigfile/1주_실습데이터.csv')
data = raw_data.copy()

## 1. 변수 확인 (데이터 건수, 컬럼 확인, 분석목표 정의)

## 2. EDA (이상치, 결측값 확인, 데이터 전처리, 시각화)

# 1. 데이터 확인
data.info() # 결측치 확인
data.describe() 
data.head()

# Y 변수 (타겟 변수) 분포 확인
target_dist = data['Y'].value_counts()

# 2. 고윳값(단일값) 횟수 출력
data.nunique() # -> x4랑 x13이 값이 하나임을 확인

data['X13']
data['X4']
# -> 둘 다 제거

data = data.drop(columns=['X4', 'X13'])


# 3. 같은 정보를 가진 변수 확인 = 단일값 찾기 -> 있다면 삭제 (동일한 값을 가진 열 찾기)
# 동일한 값을 가진 열 쌍을 찾는 코드
identical_columns = []
columns = raw_data.columns

# 각 열을 비교하여 동일한 값을 가진 열을 찾기
for i in range(len(columns)):
    for j in range(i+1, len(columns)):
        if raw_data[columns[i]].equals(raw_data[columns[j]]):
            identical_columns.append((columns[i], columns[j]))

# 결과 출력
identical_columns
# [('X6', 'X20'), ('X8', 'X18'), ('X12', 'X19')]
# 이런 경우에는 drop으로 둘 중에 하나만 제거 (x8, x6, x12 남기기)
data = data.drop(columns=['X6', 'X8', 'X12'])


# 4. 이상치 확인 (Box Plot) 하지만 그냥 안고 가기 & Box plot도 그리면 좋을듯
# IQR 방식으로 이상치 탐지
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

# 이상치 탐지
outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
print("각 변수에서 탐지된 이상치 개수:")
print(outliers)

# 각 변수에 대해 박스플롯 그리기
for col in data.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# 시각화를 보고 이상치 몇 개를 날릴지 기준을 나누기 0.1% 0.2% .. 날려보고 성능 평가 (XGB, Light)
# 0부터 1까지 0.1로 해서 이상치 날리는 조건을 하나 설정

# 대웅오빠코드 (이상치 조건)
# X4와 X13 제거 (단일값)

# 이상치 비율 리스트 (0.1% ~ 1.0%)
outlier_percents = np.arange(0.001, 0.011, 0.001)

# 결과 저장을 위한 딕셔너리
outliers_detected = {}

# 이상치 탐지 및 박스플롯 그리기
for percent in outlier_percents:
    print(f"\n이상치 비율 기준: {percent*100}%")
    
    # 상위 percent 이상치를 탐지하는 기준 설정
    upper_bound = data.quantile(1 - percent)
    lower_bound = data.quantile(percent)
    
    # 이상치 탐지 (상위, 하위 percent에 해당하는 이상치)
    is_outlier = (data < lower_bound) | (data > upper_bound)
    outliers_count = is_outlier.sum()
    
    # 이상치 탐지 결과 저장
    outliers_detected[percent] = outliers_count
    
    # 결과 출력
    print(f"각 변수에서 탐지된 이상치 개수 ({percent*100}% 기준):")
    print(outliers_count)
    
    # 박스플롯 그리기
    data.plot(kind='box', subplots=True, layout=(5, 5), figsize=(20, 15), sharex=False, sharey=False)
    plt.suptitle(f"Boxplot for {percent*100}% Outlier Removal", size=16)
    plt.show()

# 최종 결과 출력 (모든 이상치 탐지 결과)
for percent, counts in outliers_detected.items():
    print(f"\n{percent*100}% 이상치 기준:")
    print(counts)















# ----------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# 데이터 로드 및 전처리
data = pd.read_csv('../../bigfile/1주_실습데이터.csv')

# 변수 제거 (단일 값 및 동일한 값을 가진 변수 제거)
data = data.drop(columns=['X4', 'X13', 'X6', 'X8', 'X12'])

# X와 Y 분리
X = data.drop(columns=['Y'])
y = data['Y']

# 정규화 함수 정의 (MinMaxScaler 사용)
def normalize_data(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)

# 이상치 제거 및 모델 학습 함수 정의
def train_xgboost_with_outlier_removal(X, y, percent):
    # 상위, 하위 percent 이상치 탐지
    upper_bound = X.quantile(1 - percent)
    lower_bound = X.quantile(percent)
    
    # 이상치 제거
    is_outlier = (X < lower_bound) | (X > upper_bound)
    outliers_idx = is_outlier.any(axis=1)
    X_clean = X[~outliers_idx]
    y_clean = y[~outliers_idx]
    
    # 데이터 정규화
    X_clean_normalized = normalize_data(X_clean)
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X_clean_normalized, y_clean, test_size=0.2, random_state=42)
    
    # XGBoost 모델 학습
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    # 예측 및 성능 평가 (AUROC)
    y_prob = model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_prob)
    
    print(f"이상치 {percent*100}% 제거 후 AUROC: {auroc:.4f}")
    return auroc

# 이상치 비율 리스트 (0.1% ~ 1.0%)
outlier_percents = np.arange(0.001, 0.011, 0.001)

# 결과 저장을 위한 딕셔너리
results = {}

# 이상치 제거 후 XGBoost 모델 학습 및 평가
for percent in outlier_percents:
    print(f"\n이상치 비율 기준: {percent*100}%")
    auroc = train_xgboost_with_outlier_removal(X, y, percent)
    results[percent] = auroc

# 최종 결과 출력
for percent, auroc in results.items():
    print(f"\n{percent*100}% 이상치 제거 후 AUROC: {auroc:.4f}")




# ----------------------------------------------------
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# LightGBM 모델 학습 및 평가 함수 정의
def train_lightgbm(X_train, X_test, y_train, y_test):
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_prob)
    return auroc

# XGBoost 모델 학습 및 평가 함수 정의
def train_xgboost(X_train, X_test, y_train, y_test):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_prob)
    return auroc

# 데이터 정규화 함수
def normalize_data(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)

# 이상치 제거 및 두 모델 비교 함수
def compare_models(X, y, percent):
    # 상위, 하위 percent 이상치 탐지
    upper_bound = X.quantile(1 - percent)
    lower_bound = X.quantile(percent)
    
    # 이상치 제거
    is_outlier = (X < lower_bound) | (X > upper_bound)
    outliers_idx = is_outlier.any(axis=1)
    X_clean = X[~outliers_idx]
    y_clean = y[~outliers_idx]
    
    # 데이터 정규화
    X_clean_normalized = normalize_data(X_clean)
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X_clean_normalized, y_clean, test_size=0.2, random_state=42)
    
    # 두 모델의 AUROC 비교
    auroc_lightgbm = train_lightgbm(X_train, X_test, y_train, y_test)
    auroc_xgboost = train_xgboost(X_train, X_test, y_train, y_test)
    
    return auroc_lightgbm, auroc_xgboost

# 이상치 비율 리스트 (0.1% ~ 1.0%)
outlier_percents = np.arange(0.001, 0.011, 0.001)

# 결과 저장을 위한 딕셔너리
comparison_results = {}

# 이상치 제거 후 LightGBM과 XGBoost 모델 학습 및 평가
for percent in outlier_percents:
    print(f"\n이상치 비율 기준: {percent*100}%")
    auroc_lightgbm, auroc_xgboost = compare_models(X, y, percent)
    comparison_results[percent] = {'LightGBM': auroc_lightgbm, 'XGBoost': auroc_xgboost}
    print(f"LightGBM AUROC: {auroc_lightgbm:.4f}, XGBoost AUROC: {auroc_xgboost:.4f}")

# 최종 결과 출력
for percent, result in comparison_results.items():
    print(f"\n{percent*100}% 이상치 제거 후:")
    print(f"LightGBM AUROC: {result['LightGBM']:.4f}, XGBoost AUROC: {result['XGBoost']:.4f}")
# ----------------------------------------------------






















# --------------------------------------
# 5. 고유값 개수를 기준으로 숫자형인지 범주형인지 추측
# 각 변수가 범주형인지 숫자형인지 분류하고 그에 맞는 전처리 방법 적용 가능

# 고유값이 적은 변수는 범주형(Categorical)으로 간주할 수 있음
# ex. 고유값이 2~9개 정도인 변수는 클래스나 상태를 나타내는 경우 다수
# -> 범주형 변수는 One-Hot Encoding 전처리 필요할 수 있음

# 고유값이 많은 변수는 숫자형(Numerical)으로 간주할 수 있음
# -> 스케일링(정규화/표준화)가 더 유용할 가능성 높음
# -> 상관관계 분석도 가능

unique_values = data.nunique()

# 각 변수별 고유값 개수 확인
print("각 변수별 고유값 개수:")
print(unique_values)

# 범주형으로 추정되는 변수 목록 (고유값이 10개 미만인 경우 범주형으로 간주할 수 있음)
categorical_columns = unique_values[unique_values < 10].index.tolist()
print(f"범주형으로 추정되는 변수: {categorical_columns}")

# 숫자형으로 추정되는 변수 목록 (고유값이 10개 이상인 경우 숫자형으로 간주)
numerical_columns = unique_values[unique_values >= 10].index.tolist()
print(f"숫자형으로 추정되는 변수: {numerical_columns}")

# 하지만 고유값 10개를 기준으로 했을 때는 y만 범주형이고 x 변수들은 모두 숫자형으로 나옴

# 데이터 수가 엄청 많은데 범주형인지 아닌지 판단할 기준을 어떻게 세우는지?
# Categorical로 변환해서

data['X9']

## 6. 히스토그램
# 범주형 변수 시각화 (막대그래프)
for col in categorical_columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=data[col])
    plt.title(f'Countplot of {col}')
    plt.show()

# y클래스 불균형 문제로 인해 모델이 주로 **합격(0)**으로 예측하는 경향이 있을 수 있다
# 이 상황에서는 샘플링을 고려 (시간부족이슈 ,, ^^ 하지마)
# -> oversampling/ random하게 뽑는 등 방법 2~3개 정도 해보기

# 숫자형 변수 시각화 (히스토그램)
plt.figure(figsize=(12, len(numerical_columns) * 3))

for i, column in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns)//2 + len(numerical_columns)%2, 2, i)
    sns.histplot(data[column], bins=50, kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# 6. 데이터 정규화 Normalizer (변수 전체 다 이용) 해보고 변화 없으면 그냥 없다고 적어두기

# min-max랑 z-score은 별 차이 없는 것 같아 보였는데 확인 필요

# 6.1 RobustScaler 사용 (이상치(outliers)의 영향을 덜 받는 방식)
from sklearn.preprocessing import RobustScaler

# 'Y' 변수를 제외한 'X' 변수만 스케일링
x_columns = data.columns.difference(['Y'])
scaler_robust = RobustScaler()

# RobustScaler 적용
data[x_columns] = scaler_robust.fit_transform(data[x_columns])

# 결과 확인
print(data.head())

# 6.2 box-cox 사용 -  양수 데이터에 대해 데이터의 왜도를 줄여 정규분포에 가깝게 만들기 위해 사용
from sklearn.preprocessing import PowerTransformer
# 'Y' 변수를 제외한 'X' 변수만 정규화
x_columns = data.columns.difference(['Y'])

# Box-Cox 변환 적용
scaler_boxcox = PowerTransformer(method='box-cox')

# 변환 적용
data[x_columns] = scaler_boxcox.fit_transform(data[x_columns])

# 결과 확인
print(data.head())

# 6.3 log 사용 - 큰 값을 줄이고 작은 값을 확대하여 비대칭 분포를 처리하기 위해 사용
# 'Y' 변수를 제외한 'X' 변수만 로그 변환
x_columns = data.columns.difference(['Y'])

# 로그 변환 (데이터에 음수 또는 0이 있으면 1을 더해 양수로 만든 후 적용)
data[x_columns] = np.log1p(data[x_columns])  # log1p는 log(1 + x) 변환으로, 음수 값 방지

# 결과 확인
print(data.head())

# 7. 표준화를 진행해야 할까?
# ChatGPT왈:
# describe 확인 결과, 대부분의 변수는 0에서 0.69 사이에 분포 (변수 간의 범위 차이가 크지 않음을 확인)
# 표준편차도 작은 편이므로 값의 분포가 비교적 좁게 퍼져 있음
# -> 이 데이터는 표준화(Z-Score)를 반드시 적용할 필요는 없어 보임

# 8. 산점도, 산점도행렬, 상관관계 분석 Linear correlation (산점도, 히트맵, 페어플롯)
# 8.1 산점도 (X 변수와 Y 변수 간의 관계)
import matplotlib.pyplot as plt
import seaborn as sns

# 'Y' 변수와 숫자형 변수 간의 산점도
for col in numerical_columns:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=data[col], y=data['Y'])
    plt.title(f'Scatter Plot between {col} and Y')
    plt.xlabel(col)
    plt.ylabel('Y')
    plt.show()

# 8.2 산점도 행렬 (pairplot)
# 숫자형 변수와 타겟 변수 'Y'를 포함하여 산점도 행렬 그리기
# sns.pairplot(data[numerical_columns + ['Y']], diag_kind='kde', markers='o')
# plt.show()
# -> 근데 이게 너무 오래 걸려서

# 샘플링 - 데이터의 일부(예: 1000개 행)만 사용하여 산점도 행렬 그리기
sampled_data = data.sample(n=1000, random_state=42)

# 산점도 행렬 그리기 (페어플롯)
sns.pairplot(sampled_data[numerical_columns + ['Y']], diag_kind='kde', markers='o')
plt.show()

# 8.3 히트맵
# 상관관계 계산
correlation_matrix = data.corr()

# 상관관계 히트맵 그리기
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# 불량이 있고 합격이 있다면 상관관계를 잘 보여주는 변수라면 양쪽에 모여있을것
# 산점도를 찍으면 그걸 알 수 있으니까 중요한 변수겠거니 설명해주고
# 변수를 추려서 모델 학습 시작 ~!

# 샘플링된 데이터로 상관관계 분석
correlation_matrix_sampled = sampled_data.corr()

# 상관관계 히트맵 그리기
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix_sampled, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap (Sampled Data)')
plt.show()

# 타겟 변수 'Y'와의 상관관계 추출
correlation_with_target_sampled = correlation_matrix_sampled['Y'].drop('Y').sort_values(ascending=False)

# 상관관계가 높은 변수 출력 (임계값을 설정해 상관계수가 높은 변수들 선택)
threshold = 0.1  # 임계값, 원하는 대로 조정 가능
important_variables_sampled = correlation_with_target_sampled[correlation_with_target_sampled.abs() > threshold]

print("타겟 변수(Y)와 상관관계가 높은 변수들 (샘플링된 데이터):")
print(important_variables_sampled)