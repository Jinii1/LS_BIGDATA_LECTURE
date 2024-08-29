import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge 
from sklearn.preprocessing import StandardScaler 

# 블루베리 데이터 셋에 라쏘, 릿지, KNN 회귀분석 코드 적용해서
# hyper parameter 결정 후
# 각 모델의 예측값을 계산, bagging 적용해서 submit

## 필요한 데이터 불러오기
berry_train=pd.read_csv("../data/blueberry/train.csv")
berry_test=pd.read_csv("../data/blueberry/test.csv")
sub_df=pd.read_csv("../data/blueberry/sample_submission.csv")

berry_train.isna().sum()
berry_test.isna().sum()

berry_train.info()

## train
X=berry_train.drop(["yield", "id"], axis=1)
y=berry_train["yield"]
berry_test=berry_test.drop(["id"], axis=1)

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_X_scaled=scaler.transform(berry_test)

# 정규화된 데이터를 DataFrame으로 변환
X = pd.DataFrame(X_scaled, columns=X.columns)
test_X= pd.DataFrame(test_X_scaled, columns=berry_test.columns)

polynomial_transformer=PolynomialFeatures(3)

polynomial_features=polynomial_transformer.fit_transform(X.values)
features=polynomial_transformer.get_feature_names_out(X.columns)
X=pd.DataFrame(polynomial_features,columns=features)

polynomial_features=polynomial_transformer.fit_transform(test_X.values)
features=polynomial_transformer.get_feature_names_out(test_X.columns)
test_X=pd.DataFrame(polynomial_features,columns=features)

#######alpha
# 교차 검증 설정
kf = KFold(n_splits=20, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X, y, cv = kf,
                                     n_jobs = -1, scoring = "neg_mean_squared_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(2, 4, 1)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()


### model
model= Lasso(alpha=2.9)

# 모델 학습
model.fit(X, y)  # 자동으로 기울기, 절편 값을 구해줌

pred_y_lasso=model.predict(test_X) # test 셋에 대한 집값
kk





# ridge==========================================================================
# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(73, 74, 0.01)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    ridge = Ridge(alpha=alpha)
    mean_scores[k] = rmse(ridge)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

model= Ridge(alpha=73.43000000000022)

# 모델 학습
model.fit(X, y)  # 자동으로 기울기, 절편 값을 구해줌

pred_y_ridge=model.predict(test_X) # test 셋에 대한 집값
pred_y_ridge

sub_df["yield"] = pred_y_ridge
sub_df
# csv 파일로 내보내기
sub_df.to_csv("../data/blueberry/sample_submission_ridge.csv", index=False)





























train.info()
train.describe()
train.shape
test.shape

# NaN 채우기
train.isna().sum()
test.isna().sum()

x = train.drop(["yield", "id"], axis = 1)
y = train['yield']

kf = KFold(n_splits=20, shuffle=True, random_state=2024)
# ======================================================================
# Lasso
def rmse(model):
    score = np.sqrt(-cross_val_score(model, x, y, cv = kf,
                                     n_jobs = -1, scoring = "neg_mean_squared_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(100, 200, 1)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df_result = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

df_result

# 최적의 alpha 값 찾기: 
optimal_alpha = df_result['lambda'][np.argmin(df_result['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# 라쏘 회귀 모델 생성
model = Lasso(alpha = 0.87)

# 모델 학습
model.fit(x, y) 

pred_y = model.predict(test_x)

# yield 바꿔치기
sub_df["yield"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("../data/blueberry/sample_submission2.csv", index=False)
# ======================================================================
# Ridge
def rmse(model):
    score = np.sqrt(-cross_val_score(model, x, y, cv=kf,
                                     n_jobs=-1, scoring="neg_mean_squared_error").mean())
    return score

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 10, 0.01)
mean_scores = np.zeros(len(alpha_values))

k = 0
for alpha in alpha_values:
    ridge = Ridge(alpha=alpha)
    mean_scores[k] = rmse(ridge)
    k += 1

# 결과를 DataFrame으로 저장
df_result_ridge = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 최적의 alpha 값 찾기
optimal_alpha_ridge = df_result_ridge['lambda'][np.argmin(df_result_ridge['validation_error'])]
print("Optimal lambda for Ridge:", optimal_alpha_ridge)

# 릿지 회귀 모델 생성
model_ridge = Ridge(alpha=optimal_alpha_ridge)

# 모델 학습
model_ridge.fit(x, y)

# 테스트 데이터 예측
pred_y_ridge = model_ridge.predict(test.iloc[:, 1:])

# yield 바꿔치기
sub_df["yield"] = pred_y_ridge
sub_df

# csv 파일로 내보내기
sub_df.to_csv("../data/blueberry/sample_submission3.csv", index=False)
# ===============================================================================
# 단순하게 3개 y값 더해서 평균
df1=pd.read_csv("../data/blueberry/sample_submission1.csv")
df2=pd.read_csv("../data/blueberry/sample_submission2.csv")
df3=pd.read_csv("../data/blueberry/sample_submission3.csv")

df1['yield']
df2['yield']
df3['yield']

# yield 바꿔치기
sub_df["yield"] = pred_y_total=(df1['yield']+df2['yield']+df3['yield'])/3
sub_df

# csv 파일로 내보내기
sub_df.to_csv("../data/blueberry/sample_submission4.csv", index=False)
# ===============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

## 필요한 데이터 불러오기
berry_train=pd.read_csv("../../data/blueberry/train.csv")
berry_test=pd.read_csv("../../data/blueberry/test.csv")
sub_df=pd.read_csv("../../data/blueberry/sample_submission.csv")

berry_train.isna().sum()
berry_test.isna().sum()

berry_train.info()

## train
X=berry_train.drop(["yield", "id"], axis=1)
y=berry_train["yield"]
berry_test=berry_test.drop(["id"], axis=1)

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_X_scaled=scaler.transform(berry_test)

# 정규화된 데이터를 DataFrame으로 변환
X = pd.DataFrame(X_scaled, columns=X.columns)
test_X= pd.DataFrame(test_X_scaled, columns=berry_test.columns)

polynomial_transformer=PolynomialFeatures(3)

polynomial_features=polynomial_transformer.fit_transform(X.values)
features=polynomial_transformer.get_feature_names_out(X.columns)
X=pd.DataFrame(polynomial_features,columns=features)

polynomial_features=polynomial_transformer.fit_transform(test_X.values)
features=polynomial_transformer.get_feature_names_out(test_X.columns)
test_X=pd.DataFrame(polynomial_features,columns=features)

#######alpha
# 교차 검증 설정
kf = KFold(n_splits=20, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X, y, cv = kf,
                                     n_jobs = -1, scoring = "neg_mean_squared_error").mean())
    return(score)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(2, 4, 1)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()


### model
model= Lasso(alpha=2.9)

# 모델 학습
model.fit(X, y)  # 자동으로 기울기, 절편 값을 구해줌

pred_y=model.predict(test_X) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["yield"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("../data/blueberry/sample_submission_scaled.csv", index=False)