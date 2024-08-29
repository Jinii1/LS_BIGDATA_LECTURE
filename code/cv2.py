import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

# Lasso 회귀 모델의 최적의 alpha 하이퍼파라미터 값을 찾고
# 그에 따른 모델의 성능을 평가하는 과정을 구현

# 데이터 생성
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8) # -4부터 4까지 30개의 랜덤 숫자 균등하게 생성
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3) # norm.rvs라는 작은 오차를 더함

# 데이터를 DataFrame으로 변환하고 다항 특징 추가
x_vars = np.char.add('x', np.arange(1, 21).astype(str))
X = pd.DataFrame(x, columns=['x'])
# x의 1차부터 20차까지의 다항식 특징을 생성하기 위해 PolynomialFeatures 객체 생성
poly = PolynomialFeatures(degree=20, include_bias=False) # 1을 붙이냐 안 붙이냐 (상수항 포함x)
# 원래의 x에서 1차부터 20차까지의 다항식 특징을 생성
# X_poly는 20개의 열을 가진 새로운 데이터프레임이 됨
X_poly = poly.fit_transform(X) # 1~20차까지의 벡터를 자동으로 계산해서 만들어줌 
X_poly=pd.DataFrame(
    data=X_poly,
    columns=x_vars
)
X_poly.shape # 30x20: 원래 30개의 데이터 X 20차까지의 poly

# 교차 검증 설정, KFold 객체 생성
# 데이터를 K개의 폴드로 나눈 후, 각 폴드에 대해 학습과 평가를 반복적으로 수행
# K개의 폴드 중 하나를 평가용 데이터로 사용, 나머지 K-1개의 폴드를 학습용 데이터로 사용
# 데이터를 3등분하고, 각 분할마다 학습과 검증을 반복해서 모델 성능을 평가하기 위해 사용
# shuffle=True는 데이터를 무작위로 섞어준다는 의미
kf = KFold(n_splits=3, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, X_poly, y, cv = kf,
                                     n_jobs=-1, scoring = "neg_mean_squared_error").mean())
    return(score)
# cross_val_score: 교차모델을 수행하여 각 폴드 별 성능 점수 반환
# scoring='neg_MSE': 성능평가지수=MSE(평균제곱오차), score 높을수록 성능 좋기 때문에 오차값을 음수로 반환하여 이를 뒤집음
# .mean()함으로써 전체성능
# return 값이 작을수록 예측값이 실제값과 비슷

lasso = Lasso(alpha=0.01)
ridge = Ridge(alpha=0.01)
rmse(lasso)

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 10, 0.01) # alpha값: lasso나 ridge 모델의 hyperparameter, 모델 훈련할 때 규제 강도 조절
mean_scores = np.zeros(len(alpha_values)) # 각 alpha 값에 대한 모델의 평균 성능 점수를 저장할 배열

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
df

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
# np.argmin: 주어진 배열이나 시리즈에서 가장 작은 값의 인덱스를 반환
print("Optimal lambda:", optimal_alpha)