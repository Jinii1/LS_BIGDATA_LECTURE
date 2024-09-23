import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
import matplotlib.pyplot as plt
import seaborn as sns

penguins = load_penguins()
penguins.head()

# 펭귄 분류 문제
# y: 펭귄의 종류
# x1: bill_length_mm (부리 길이)
# x2: bill_depth_mm (부리 깊이)

df=penguins.dropna()
df=df[['species', "bill_length_mm", "bill_depth_mm"]]
df=df.rename(columns={'species': 'y',
                      'bill_length_mm': 'x1',
                      'bill_depth_mm': 'x2'})
df

# x1, x2의 산점도를 그리되, 점 색깔은 펭귄 종별 다르게
df.plot(kind='scatter', x='x1', y='x2', color='y')
sns.scatterplot(data = df, x = 'x1', y = 'x2', hue = "y")
plt.axvline(x=45)
# -> 왼쪽엔 Adelie, 오른쪽엔 Gentoo&Chinstrap

# Q. 나누기 전 현재의 엔트로피는?
# Q. 45로 나눴을 때 엔트로피 평균은 얼마인가요?
# 엔트로피: 데이터가 얼마나 무질서한지 (혼란스러운지)
# 데이터를 특정 기준으로 나눴을 때, 나누기 전과 후의 엔트로피가 얼마나 줄어드는지 계산하는건 분류에서 중요

# 엔트로피 구하려면 각 종별 갯수 구하면 됨
df['y'].value_counts()
# p1, p2, p3구하기
p_i=df['y'].value_counts() / len(df['y'])
entropy_curr=sum(p_i * np.log2(p_i))

# x1=45 기준으로 나눈 후, 평균 엔트로피
# x1=45 기준으로 나눴을 때, 데이터 포인트가 몇 개씩 나뉘는지
n1=df.query('x1< 45').shape[0] # 1번 그룹
n2=df.query('x1>= 45').shape[0] # 2번 그룹

# 1번 그룹은 어떤 종류로 예측하나요?
# 2번 그룹은 어떤 종류로 예측하나요?
y_hat1=df.query('x1 < 45')['y'].mode()
y_hat2=df.query('x1 >= 45')['y'].mode()

# 각 그룹 엔트로피는 얼마인가요?
p_1=df.query('x1 < 45')['y'].value_counts() / len(df.query('x1 < 45')['y'])
entropy1=sum(p_1 * np.log2(p_1))

p_2=df.query('x1 >= 45')['y'].value_counts() / len(df.query('x1 >= 45')['y'])
entropy2=sum(p_2 * np.log2(p_2))

# 평균 엔트로피 구하기
# 나누기 전과 후 엔트로피 비교하고자 함
# 두 그룹의 엔트로피를 가중평균으로 계산함
# -> 그룹의 크기 n1, n2에 따라 각각 엔트로피가 얼마나 큰 지 비중을 두고 합친 값
entropy_x145=(n1 * entropy1 + n2 * entropy2) / (n1+n2)
# entropy_curr보다 값이 줄어듦 -> 무질서도 감소
# 이 분할로 인해 더 예측 가능하고 잘 분류된 상태, 즉 좋은 기준이 될 수 있다는 결론
# -> 정보 이득 (Information Gain)
# ====================================================
# Q1. entropy 만들고 x1 기준으로 최적 기준값은 얼마인가?
# Q2. 기준값 x를 넣으면 entropy값이 나오는 함수는?

# 최적 기준값을 구하는이유? 데이터 가장 잘 분리하는 기준 찾기 위해서
# -> 분류 성능 높이고 무질서도 최소화 -> 데이터 잘 예측

# 기준값 x를 넣으면 entropy값이 나오는 함수는?
def my_entropy(x): 
    # x기준으로 나눴을때, 데이터 포인터가 몇개 씩 나뉘나요?
    n1 = df.query(f"x1 < {x}").shape[0] # 1번 그룹
    n2 = df.query(f"x1 >= {x}").shape[0] # 2번 그룹
    # 각 그룹 엔트로피는 얼마 인가요?
    p_1 = df.query(f"x1 < {x}")['y'].value_counts() / len(df.query(f"x1 < {x}")['y'])
    entropy1 = -sum(p_1 * np.log2(p_1))
    p_2 = df.query(f"x1 >= {x}")['y'].value_counts() / len(df.query(f"x1 >= {x}")['y'])
    entropy2 = -sum(p_2 * np.log2(p_2))
    entropy_x = (n1*entropy1 + n2*entropy2) / (n1+n2)
    return(entropy_x)

my_entropy(45)

# x1 기준으로 최적 기준값은 얼마인가?

# entropy계산을 해서 가장 작은 entropy가 나오는 x는?
import numpy as np
from scipy import optimize

entropy_list = []
x_list = np.arange(df["x1"].min(), df["x1"].max(), 0.01)

for i in x_list : 
    entropy_list.append(my_entropy(i))

entropy_list

# entropy_list 최소값
min(entropy_list)

# entropy_list를 최소로 만드는 x의 값
x_list[np.argmin(entropy_list)]
