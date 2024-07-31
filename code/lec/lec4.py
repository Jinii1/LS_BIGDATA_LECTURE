a=(1,2,3)
a

a=[1,2,3]
a

# soft copy
b= a
a

a[1]=4
a

b
id(a)
id(b)

# deep copy
a=[1,2,3]
a

b=a[:] # method 1
b=a.copy() # method 2

a[1]=4
a
b

# 수학 함수 요약 표
math.sqrt()
math.exp()
math.log(x,[base])
math.factorial()
math.sin()
math.cos()
math.sin()
math.radians()
math.degrees()

# 수학함수 예제
import math
x=4
math.sqrt(x)

exp_val = math.exp(5)
exp_val

log_val = math.log(10,10)
log_val

fact_val = math.factorial(5)
fact_val

sin_val = math.sin(math.radians(90))
sin_val
cos_val = math.cos(math.radians(90))
cos_val
tan_val = math.tan(math.radians(90))
tan_val

# 예제 1
import numpy as np
import math
def a(x, mu, sigma):
  
  part_1 = (sigma * math.sqrt(2 * math.pi)) ** -1
  part_2 = math.exp((-(x-mu)**2) /(2*sigma**2))
  return part_1 * part_2

a(1, 0, 1)

# 예제 2
def b(x,y,z):
    return (x**2 + math.sqrt(y) + math.sin(z)) * math.exp(x)
b(2,9,math.pi/2)

# 예제 3
def c(x):
    return math.cos(x) + (math.sin(x) * math.exp(x))
c(math.pi)

# fcn + tab                 
def     (input):
    contents
    return 

# ! pip install numpy
# Ctrl + Shift + C: 커멘트 처리

import numpy as np
    
  # 벡터 생성하기 예제
a = np.array([1, 2, 3, 4, 5]) # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"]) # 문자형 벡터 생성
c = np.array([True, False, True, True]) # 논리형 벡터 생성
print("Numeric Vector:", a)
print("String Vector:", b)
print("Boolean Vector:", c)
  
a
type(a)  
a[3]
a[2:]
a[1:4]

# 빈 배열 생성
b = np.empty(3)
b[0] = 1
b[1] = 4
b[2] = 10
b
b[2]

vec1 = np.array([1,2,3,4,5])
vec1 = np.arange(100)
vec1 = np.arange(1,100.5, 0.5)
vec1

l_space1 = np.linspace(0, 1, 5) # np.linspace 함수: 시작점과 종료점 사이에서 균일한 간격의 숫자 배열 생성
print("0부터 1까지 5개 원소:", l_space1)

# -100부터 0까지
vec2= np.arange(-100,1)
vec2
vec3 = np.arange(0,-100,-1)
vec3

l_space2 = np.linspace(0, 1, 5, endpoint=False) # endpoint 옵션: 해당 값은 포함 x
l_space2

# repeat vs tile 함수
vec1 = np.arange(5)
np.repeat(vec1,3) #array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
np.tile(vec1,3) #array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
vec1 * 6 #array([ 0,  6, 12, 18, 24])

vec1=np.array([1,2,3,4])
vec1+vec1

max(vec1)

# 35672 이하 홀수들의 합은?
x = np.arange(1,35673,2) # method1
sum(x)

np.arange(1,35673,2).sum() # method2

# 넘파이 벡터 길이 재는 법
len(x)
x.shape

[[1,2,3],[4,5,6]]
b=np.array([[1,2,3],[4,5,6]])

# 다차원 배열의 길이 재기 - 2차원 배열
b = np.array([[1, 2, 3], [4, 5, 6]])
length = len(b) # 첫 번째 차원의 길이
shape = b.shape # 각 차원의 크기
size = b.size # 전체 요소의 개수
length, shape, size

import numpy as np
a=np.array([1,2])
b=np.array([1,2,3,4])
a+b
np.repeat(a,2)+b

b == 3 # b안에 3이 있는지 없는지

# 35672 보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 갯수는?
sum((np.arange(1,35672) % 7) == 3)

# 10 보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 갯수는?
sum((np.arange(1,10) % 7) == 3)

a = np.array([1.0, 2.0, 3.0])
b = 2.0
a * b

a.shape # shape 존재 o
b.shape # shape 존재 x

# 2차원 배열 생성
matrix = np.array([[ 0.0, 0.0, 0.0],
                    [10.0, 10.0, 10.0],
                    [20.0, 20.0, 20.0],
                    [30.0, 30.0, 30.0]])
matrix.shape
# 1차원 배열 생성
vector = np.array([1.0, 2.0, 3.0, 4.0])
vector.shape # 값이 하나일 경우에 튜플로 불러올 때 (숫자,)로 되어야 함

# 브로드캐스팅을 이용한 배열 덧셈
# 길이가 다른 배열 간의 연산을 가능하게 해주는 메커니즘
result = matrix + vector
print("브로드캐스팅 결과:\n", result)

vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1)
vector
vector.shape
result = matrix + vector
result


















