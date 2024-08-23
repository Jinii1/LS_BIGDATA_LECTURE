import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


x = np.linspace(-100, 100, 400)
y = 2*(x**2) + 3*x + 5

plt.plot(x, y, color = 'pink')
plt.axvline(x=0, color = 'black')

y = -3*x**3 + 3*x**2 + 5*x - 1
plt.plot(x, y, color = 'pink')
plt.axvline(x=0, color = 'black')
plt.axhline(y=0, color = 'black')

# 데이터 만들기
from scipy.stats import norm
from scipy.stats import uniform

# 검정곡선 
k = np.linspace(-4, 4, 200)
sin_y = np.sin(k)

# 파란 점들 
x = uniform.rvs(size = 20, loc= -4, scale = 8)
y = np.sin(x) + norm.rvs(size=20, loc=0, scale = 0.3)  # epsilon (noise)

plt.plot(k, sin_y, color= "black")
plt.scatter(x, y, color = "blue")

# 파란 점을 가지고 어떻게 추정을 할 것인가가 목표임. 

np.random.seed(42)
x = uniform.rvs(size = 30, loc = -4, scale = 8)
y = np.sin(x) + norm.rvs(size =30, loc=0, scale = 0.3)
df = pd.DataFrame({'x': x, 
                   'y': y})
train_df = df.loc[:19]
test_df = df.loc[20:]

plt.scatter(data=train_df, x = 'x', y = 'y', color = "blue")

from sklearn.linear_model import LinearRegression
model = LinearRegression()
x = train_df[['x']]
y = train_df['y']

model.fit(x,y)
model.coef_
model.intercept_

reg_line = model.predict(x)

plt.scatter(data=train_df, x = 'x', y = 'y', color = "blue")
plt.plot(x, reg_line, color = "red")

# 2차 곡선 회귀
train_df['x2'] = train_df['x']**2

x = train_df[["x", "x2"]]
y = train_df["y"]
model.fit(x, y)

model.coef_
model.intercept_


k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x": k, "x2": k**2
})
reg_line = model.predict(df_k)

plt.scatter(data=train_df, x = 'x', y = 'y', color = "blue")
plt.plot(k, reg_line, color = "red")


# 3차 곡선 회귀
train_df['x2'] = train_df['x']**2
train_df['x3'] = train_df['x']**3

x = train_df[["x", "x2", "x3"]]
y = train_df["y"]
model.fit(x, y)

model.coef_
model.intercept_


k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x": k, "x2": k**2, "x3": k**3
})
reg_line = model.predict(df_k)

plt.scatter(data=train_df, x = 'x', y = 'y', color = "blue")
plt.plot(k, reg_line, color = "red")


# 4차 곡선 회귀
train_df['x4'] = train_df['x']**4


x = train_df[["x", "x2", "x3", "x4"]]
y = train_df["y"]
model.fit(x, y)

model.coef_
model.intercept_


k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x": k, "x2": k**2, "x3": k**3, "x4": k**4
})
reg_line = model.predict(df_k)

plt.scatter(data=train_df, x = 'x', y = 'y', color = "blue")
plt.plot(k, reg_line, color = "red")



# 20차 곡선 회귀
train_df['x5'] = train_df['x']**5
train_df['x6'] = train_df['x']**6
train_df['x7'] = train_df['x']**7
train_df['x8'] = train_df['x']**8
train_df['x9'] = train_df['x']**9
train_df['x10'] = train_df['x']**10
train_df['x11'] = train_df['x']**11
train_df['x12'] = train_df['x']**12
train_df['x13'] = train_df['x']**13
train_df['x14'] = train_df['x']**14
train_df['x15'] = train_df['x']**15
train_df['x16'] = train_df['x']**16
train_df['x17'] = train_df['x']**17
train_df['x18'] = train_df['x']**18
train_df['x19'] = train_df['x']**19
train_df['x20'] = train_df['x']**20


x = train_df[["x", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
              "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18",
              "x19", "x20"]]
y = train_df["y"]
model.fit(x, y)

model.coef_
model.intercept_


k = np.linspace(-4, 4, 200)
df_k = pd.DataFrame({
    "x": k, "x2": k**2, "x3": k**3, "x4": k**4,
    "x5": k**5, "x6": k**6, "x7": k**7, "x8": k**8, "x9": k**9
})
reg_line = model.predict(df_k)

plt.scatter(data=train_df, x = 'x', y = 'y', color = "blue")
plt.plot(k, reg_line, color = "red")

# test
test_df = df.loc[20:]
test_df['x2'] = test_df['x']**2
test_df['x3'] = test_df['x']**3
test_df['x4'] = test_df['x']**4
test_df['x5'] = test_df['x']**5
test_df['x6'] = test_df['x']**6
test_df['x7'] = test_df['x']**7
test_df['x8'] = test_df['x']**8
test_df['x9'] = test_df['x']**9
test_df['x10'] = test_df['x']**10
test_df['x11'] = test_df['x']**11
test_df['x12'] = test_df['x']**12
test_df['x13'] = test_df['x']**13
test_df['x14'] = test_df['x']**14
test_df['x15'] = test_df['x']**15
test_df['x16'] = test_df['x']**16
test_df['x17'] = test_df['x']**17
test_df['x18'] = test_df['x']**18
test_df['x19'] = test_df['x']**19
test_df['x20'] = test_df['x']**20

model.coef_
x = test_df[["x", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
              "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18",
              "x19", "x20"]]

y_hat = model.predict(x)

sum((test_df["y"] - y_hat) ** 2)