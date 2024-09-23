# 목표: 모든 변수를 넣어서 lasso regression해서 제출 
# 과정: 
# 모든 변수 전처리
# train data로 넣고 cross validation 적용해서 lambda 찾고
# test data set을 predict 해서 y 찾고 submit

# 질문:
# 주어진 데이터셋을 사용하여 Lasso 회귀 모델을 구축하고
# 최적의 하이퍼파라미터를 찾은 후
# 테스트 데이터에 대해 예측을 제출하는 과정에 대해 설명하세요
# 답변에는 다음 내용을 포함하세요:

# 데이터 전처리:
# 결측값을 처리하는 방법과 그 이유를 설명하세요.

# 피처 엔지니어링:
# 데이터에서 범주형 변수와 수치형 변수를 어떻게 변환했는지 설명하세요.

# 하이퍼파라미터 튜닝:
# Lasso 회귀 모델의 alpha 값을 어떻게 최적화했는지 설명하세요.

# 모델 평가:
# 교차 검증을 사용하여 모델 성능을 평가한 방법을 설명하세요.

# 예측 및 제출:
# 최적의 모델을 사용하여 테스트 데이터에 대한 예측을 수행하고, 결과를 제출하는 방법을 설명하세요.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

## 필요한 데이터 불러오기
house_train=pd.read_csv("../data/houseprice/train.csv")
house_test=pd.read_csv("../data/houseprice/test.csv")
sub_df=pd.read_csv("../data/houseprice/sample_submission.csv")

# NaN 채우기
house_train.isna().sum()
house_test.isna().sum()

# 각 숫자변수는 평균채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)

house_train[quant_selected].isna().sum()

# 각 변수형변수는 최빈값으로 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna(house_train[col].mode(), inplace=True)

house_train[qual_selected].isna().sum()


train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
df = pd.concat([house_train, house_test], ignore_index=True)

df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include = [object]).columns,
    drop_first=True
    )
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

X = train_df.drop(["SalePrice", "Id"], axis = 1)
y = train_df['SalePrice']

# 교차 검증 설정
kf = KFold(n_splits=20, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X, y, cv = kf,
                                     n_jobs = -1, scoring = "neg_mean_squared_error").mean())
    return(score)


# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(120, 125, 0.01)
# 최적의 알파값을 어떻게 찾나?
# 일반적으로 np.arange(0.001, 100, 0.01) 같은 범위를 사용하여 보다 넓은 범위 탐색
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

# 결과 시각화
plt.plot(df_result['lambda'], df_result['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기: 120
optimal_alpha = df_result['lambda'][np.argmin(df_result['validation_error'])]
print("Optimal lambda:", optimal_alpha)

# 라쏘 회귀 모델 생성
model = Lasso(alpha = 120)

# 모델 학습
model.fit(X, y) 

test_df=df.iloc[train_n:,]
test_df = test_df.drop(["SalePrice", "Id"], axis = 1)

quantitative = test_df.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    test_df[col].fillna(test_df[col].mean(), inplace=True)

test_df[quant_selected].isna().sum()

# 각 변수형변수는 최빈값 채우기
qualitative = test_df.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    test_df[col].fillna(test_df[col].mode(), inplace=True)

test_df[qual_selected].isna().sum()

pred_y = model.predict(test_df)

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/sample_submission3.csv", index=False)
