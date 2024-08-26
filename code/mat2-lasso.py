import numpy as np

# 회귀분석 데이터행렬
x=np.array([13, 15,
           12, 14,
           10, 11,
           5, 6]).reshape(4, 2)
x
vec1=np.repeat(1, 4).reshape(4, 1)
matX=np.hstack((vec1, x))
y=np.array([20, 19, 20, 12]).reshape(4, 1)
matX

# minimize로 라쏘 베타 구하기
from scipy.optimize import minimize

def line_perform_lasso(beta): # Lasso 회귀의 손실 함수를 정의
    beta=np.array(beta).reshape(3, 1)
    a=(y - matX @ beta)
    return (a.transpose() @ a) + 3*np.abs(beta[1:]).sum()

line_perform_lasso([8.55,  5.96, -4.38])
line_perform_lasso([8.14,  0.96, 0])

# 초기 추정값
initial_guess = [0, 0, 0]

# 최소값 찾기
result = minimize(line_perform_lasso, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)

[8.55, 5.96, -4.38] # 랃다 0일때

[8.14,  0.96, 0] # 람다 3일떄
# 예측식: y_hat=8.14 + 0.96*X1 + 0*X2

[17.74, 0, 0] # 람다가 500일때
# 예측식: y_hat=17.74 + 0*X1 + 0*X2

# 람다 값에 따라 변수 선택됨
# X 변수가 추가되면, trainX에서는 성능이 항상 좋아짐
# X 변수가 추가되면, ValidX에서는 좋아졌다가 나빠짐 (나빠지는 순간이 Overfitting)
# 어느 순간 X 변수 추가하는 것을 멈춰야 함
# 람다 0부터 시작: 내가 가진 모든 변수를 넣겠다!
# 점점 람다를 증가: 변수가 하나씩 빠지는 효과
# validX에서 가장 성능이 좋은 람다를 선택! -> 변수가 선택됨을 의미

# X의 칼럼에 선형 종속인 애들 있다: 다중공선성이 존재