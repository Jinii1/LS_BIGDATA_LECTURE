
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import boxcox

# 예제 데이터 생성 (양수 데이터)
np.random.seed(0)
data = np.random.exponential(scale=2, size=1000)

# Box-Cox 변환 전 데이터의 히스토그램
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(data, bins=30, color='blue', alpha=0.7)
plt.title('Before Box-Cox Transformation')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Box-Cox 변환
data_boxcox, best_lambda = boxcox(data)

# Box-Cox 변환 후 데이터의 히스토그램
plt.subplot(1, 2, 2)
plt.hist(data_boxcox, bins=30, color='green', alpha=0.7)
plt.title('After Box-Cox Transformation')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 최적의 람다 값 출력
print(f'Best lambda for Box-Cox transformation: {best_lambda}')
