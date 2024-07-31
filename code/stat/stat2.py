import pandas as pd
import numpy as np
import matplotlib as plt

# 0~ 1 사이 숫자 5개 생성하기
y = np.random.rand(50000).reshape(-1, 5).mean(axis=1) # method 1
y
x = np.random.rand(10000, 5) \
      .reshape(-1, 5) \
      .mean(axis =1) # method 2
x

# 정규분포 만들기
x = np.random.rand(10000, 5).mean(axis=1)
plt.hist(x, bins = 30, alpha = 0.7, color = 'blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

import numpy as np

x = np.arange(33)
sum(x)/33
sum((x - 16) * 1/33)
(x - 16) ** 2
np.unique ((x - 16) ** 2)

np.unique ((x - 16) ** 2) * (2/33)
sum(np.unique((x - 16) ** 2) * (2/33))

# E[X^2]
sum(x**2 * (1/33))

# Var(X) = E[X^2] - (E[X]^2)
sum(x**2 * (1/33)) - 16**2

## Example 1
x = np.arange(4)
x
pro_x = np.array([1/6, 2/6, 2/6, 1/6])
sum(pro_x)

# 기댓값
Ex =sum(x * pro_x) # E[X] 
Exx =sum(x**2 * pro_x) #E[X^2]

# 분산
Exx - Ex**2

sum((x - Ex) ** 2 * pro_x)

## Example 2
x=np.arange(99)
x

# 1-50-1 벡터
x_1_50_1=np.concatenate((np.arange(1, 51), np.arange(49, 0, -1)))
pro_x=x_1_50_1/2500

# 기대값
Ex=sum(x * pro_x)
Exx=sum(x**2 * pro_x)

# 분산
Exx - Ex**2
sum((x - Ex)**2 * pro_x)

sum(np.arange(50))+sum(np.arange(51))

## Example 3
Y = np.arange(4) * 2 # 난 이렇게 짬 .. Y = np.arange(0,7,2)
pro_y = np.array([1/6, 2/6, 2/6, 1/6])

Ey =sum(Y * pro_y) # E[X] 
Eyy =sum(Y**2 * pro_y) #E[X^2]

# 분산
Eyy - Ey**2

np.sqrt(9.52**2/10)
