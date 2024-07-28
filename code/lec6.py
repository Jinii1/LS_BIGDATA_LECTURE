import numpy as np
import pandas as pd

# 행렬: 길이가 같은 벡터를 사각형 모양으로 묶어 놓은
# 이 사각형의 크기는 shape 속성을 통해 측정 가능

# 두 개의 벡터를 합쳐 행렬 생성
matrix = np.column_stack((np.arange(1, 5),
np.arange(12, 16)))
type(matrix)
print("행렬:\n", matrix)

# 행렬 만들기

# np.zeros((행, 열)): 지정된 형태의 모든 요소가 0인 행렬 생성
# np.reshape((행, 열)): 배열의 형태를 지정된 행과 열로 변환
np.zeros(5)
np.zeros([5, 4])
np.arange(1, 5).reshape((2, 2))
np.arange(1, 7).reshape((2, 3))
# -1 통해서 크기를 자동으로 결정할 수 있음
np.arange(1, 7).reshape((2, -1))

# Q. 0에서부터 99까지 수 중 랜덤하게 50개 숫자를 뽑아서
# 5 by 10 행렬 만드세요.
np.random.randint(0, 100, 50).reshape((5, 10))

# 행렬을 채우는 방법 - order 옵션
# order = 'C' 행 우선 순서
np.arange(1, 21).reshape((4, 5), order = 'C')
# order = 'F' 열 우선 순서
mat_a = np.arange(1, 21).reshape((4, 5), order = 'F')

# indexing
mat_a[0, 0]
mat_a[2, 3]
mat_a[0:2, 3]
mat_a[1:3, 1:4]

# 행자리, 열자리 비어있는 경우 전체 행 또는 열 선택
mat_a[3, :]
mat_a[3, ::2]

# 짝수 행만 선택하려면?
mat_b = np.arange(1, 101).reshape((20, -1))
mat_b[1::2, :]

# 행렬 필터링
x= np.arange(1, 11).reshape((5, 2)) *2
x
x[[True, True, False, False, True], 0]

mat_b[:, 1]   # 벡터
mat_b[:, 1].reshape((-1, 1))
mat_b[:,[1]]  # 행렬
mat_b[:, 1:2] # 행렬

# 필터링
mat_b[mat_b[:, 1]%7 == 0, :]
# mat_b[:, 1]%7 == 0을 만족하는 행은 7, 42, 77이 포함된 행들 전부 출력

# 사진은 행렬이다
import numpy as np
import matplotlib.pyplot as plt

# 0 검은색, 1흰색 흑백 이미지 생성
# 난수 생성하여 3x3 크기의 행렬 생성
np.random.seed(2024)
img1 = np.random.rand(3, 3) # 0과 1사이의 난수 9개 생성
print("이미지 행렬 img1:\n", img1)
import matplotlib.pyplot as plt
plt.show()

plt.clf()

plt.imshow(img1, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()

# 행렬에서 사진으로
# 행렬 안의 값이 0과 1사이에 있어야 함, 그렇지 않다면 변환
# 변수가 0부터 9까지 있으니까 9로 나눠서 데이터를 0부터 1까지만 만듦
a = np.random.randint(0, 10, 20).reshape(4, -1)
a
a/9

a = np.random.randint(0, 256, 20).reshape(4, -1)
a
a/255

import urllib.request
img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "jelly.png")

import imageio
! pip install imageio

# 이미지 읽기
jelly = imageio.imread("img/jelly.png")
print("이미지 클래스:", type(jelly))
print("이미지 차원:", jelly.shape)
print("이미지 첫 4x4 픽셀, 첫 번째 채널:\n", jelly[:4, :4, 0])

jelly.shape # 50 x 4가 80장
jelly[:, :, 0].shape
jelly[:, :, 0].transpose().shape

plt.imshow(jelly)
plt.imshow(jelly[:, :, 0].transpose())
plt.imshow(jelly[:, :, 0]) # R
plt.imshow(jelly[:, :, 1]) # G
plt.imshow(jelly[:, :, 2]) # B
plt.imshow(jelly[:, :, 3]) # 투명도
plt.axis('off') # 축 정보 없애기
plt.show()
plt.clf()

# 3차원 배열

# 두 개의 2x3 행렬 생성
mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)

my_array = np.array([mat1, mat2])
my_array.shape

first_slice = my_array[0, :, :]

# 마지막 요소 제외
filtered_array = my_array[:, :, :-1]

my_array[:, :, [0, 2]]
my_array[:, 0, :]
my_array[0, 1, 1:3] # [0, 1, [1, 2]]

mat_x = np.arange(1, 101).reshape((5, 5, 4))
mat_y = np.arange(1, 101).reshape((10, 5, 2)) # ((-1, 5, 2))

my_array2 = np.array([my_array, my_array])
my_array2.shape

# 넘파이 배열 메서드
a = np.array([[1, 2, 3], [4, 5, 6]])

a.sum()
a.sum(axis=0)
a.sum(axis=1)

a.mean()
a.mean(axis=0)
a.mean(axis=1)

mat_b = np.random.randint(0, 100, 50).reshape((5, -1))
mat_b

# 행별로 가장 큰 수는?
mat_b.max(axis=1)

# 열별로 가장 큰 수는?
mat_b.max(axis=0)

a = np.array([1, 3, 2, 5])

# 누적 합계
mat_b.cumsum(axis=1) # 행별 누적 합계

# 누적 곱
a = np.array([1, 3, 2, 5])
a.cumprod()

# 대괄호 갯수로 (n차원 배열로) 반환
mat_b.reshape((2, 5, 5)).flatten()

# 주어진 범위로 자름 .clip(min, max)
d = np.array([1, 2, 3, 4, 5])
d.clip(2, 4)

# 리스트로 변환
d.tolist()
