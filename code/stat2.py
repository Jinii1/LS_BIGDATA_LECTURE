import numpy as np
import matplotlib.pyplot as plt

# 예제 넘파이 배열 생성
data = np.random.rand(10)

# 히스토그램
plt.hist(data, bins = 5, alpha = 1, color = 'blue') # bin 나누는 구간, alpha 투명도
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
plt.clf()

data = np.random.rand(0:1, 5)
plt.clf()
plt.hist(data, bins = 5, alpha = 1, color = 'black')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 0~ 1 사이 숫자 5개 생성하기
np.random.rand(50000).reshape(-1, 5).mean(axis=1) # method 1

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
