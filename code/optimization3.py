# ==========================
# 회귀직선 베타 찾기
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

# x, y의 값을 정의
beta0 = np.linspace(-20, 20, 400)
beta1 = np.linspace(-20, 20, 400)
beta0, beta1 = np.meshgrid(beta0, beta1)

# 함수 f(x, y)를 계산
z = (1-(beta0+beta1))**2 + (4-(beta0+2*beta1))**2 + (1.5-(beta0+3*beta1))**2 + (5-(beta0+4*beta1))**2

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(beta0, beta1, z, levels=100)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

# 특정 점 (9, 2)에 파란색 점을 표시
plt.scatter(9, 9, color='red', s=100)

beta0=9; beta1=9
lstep=0.01
for i in range(1000):
    beta0, beta1 = np.array([beta0, beta1]) - lstep * np.array([8*beta0+20*beta1-23, 20*beta0 + 60*beta1-67])
    plt.scatter(float(beta0), float(beta1), color='red', s=25)

# 계속 
# 내가 coefficence beta 찾았다는 걸 어떻게 알까?
# 이전값과 다음값을 계속 추적하고 계속 변하면 아직 수렴이 안 됐나보다
# 거의 변하지 않으면 stop할수도 있음
# -> Early Stopping 이라고 부름 (변화량이 작아질 때 반복 멈추기)

print(beta0, beta1)

# 축 레이블 및 타이틀 설정
plt.xlabel('beta0')
plt.ylabel('beta1')
plt.xlim(-10, 10)
plt.ylim(-10, 10)

# 그래프 표시
plt.show()

# 모델 fit으로 베타 구하기
df=pd.DataFrame({
    'x': np.array([1, 2, 3, 4]),
    'y': np.array([1, 4, 1.5, 5])
})
model = LinearRegression()
model.fit(df[['x']], df['y'])

model.intercept_
model.coef_
# ==============================================================
# x1이 15, x2가 5.6인 y는?
df=pd.DataFrame({
    'x1': [16, 20, 22, 18, 17],
    'x2': [7, 7, 5, 6, 7],
    'y': [13, 15, 17, 15,16]
})
df
X = df[['x1', 'x2']] 
y=df['y']



model = LinearRegression()
model.fit(X, y)

new_data = np.array([[15, 5.6]])
y_new_pred = model.predict(new_data)