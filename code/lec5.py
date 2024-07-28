import numpy as pd

# 벡터의 일부를 추출할 때는 []사용 (위치나 인덱스 지정)
# 인덱스는 0부터 시작하고 음의 인덱스는 뒤에서부터 셈
# [start, stop, step]

# 벡터 슬라이싱 예제, a를 랜덤하게 채움
np.random.seed(42)
a = np.random.randint(1, 21, 10) # 1부터 20까지의 수를 10개
# 1부터 3까지의 수를 100번 뽑는데 확률을 이렇게 두고 중복 안 되게 추출
a = np.random.choice(np.arange(1, 4), 100, True, np.array([2/5, 2/5, 1/5]))
print(a)
sum(a == 3) # 3 나오는 갯수

# 두 번째 값 추출
print(a[1])

a[2:5] 
a[-1] # 맨 끝에서 두번째 (마지막값, 음수 인덱스 사용)
a[::2] # 처음부터 끝까지, step은 2 (=두칸씩 띄우고)
a[0:6:2] # 두번째 자리부터 5까지 두칸씩 띄워서

# 1에서부터 1000사이 3의 배수의 합은?
sum(np.arange(3,1001,3)) # method 1
x = np.arange(0,1001) # method 2
sum(x[::3])

print(a[[0,2,4]])
np.delete(a,1)
np.delete(a,[1,3])

a
a>3
a[a>3]

np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
a[(a>2000) & (a<5000)]
# a[조건을 만족하는 논리형 벡터]
# 값이 true/false로 나옴

! pip install pydataset
import pydataset

df=pydataset.data('mtcars')
np_df=np.array(df['mpg'])

model_names = np.array(df.index)

# 15 이상 25 이하인 데이터 개수는?
sum ((np_df >= 15) & (np_df <= 25))

# 평균 mpg보다 높은 (이상) 자동차
sum(np_df >= np.mean(np_df))

# 15보다 작거나 22 이상인 데이터 개수는?
((np_df<=15) | (np_df >= 22))

# 15 이상 25 이하인 자동차 모델은?
model_names[(np_df >= 15) & (np_df <=20)]

# 평균 mpg보다 높은(이상) 자동차 모델은?
model_names[np_df >= np.mean(np_df)]

# 연비가 안 좋은 자동차 모델은?
model_names[np_df < np.mean(np_df)]

np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
b = np.array (['A', 'B', 'C', 'F', 'W'])
a[(a>2000) & (a<5000)]
b[(a>2000) & (a<5000)] # a의 원소로 필터링을 하는데 실제 원소는 b에서 가져온다

a[a > 3000] = 3000

np.random.seed(2024)
a = np.random.randint(1,100,10)
a < 50
np.where(a < 50) # 조건에 만족하는 원소의 위치를 반환

np.random.seed(2024)
a = np.random.randint(1, 26346, 1000)
a

# 처음으로 22000보다 큰 숫자가 나왔을 때, 숫자의 위치와 그 숫자는 무엇인가요?
x=np.where(a > 22000)
type(x)
my_index = x[0][0]
a[my_index]

# 처음으로 24000보다 큰 숫자 나왔을 때, 숫자 위치와 그 숫자는 무엇인가요?
x = np.where(a>24000)
a[x[0][0]]

# 처음으로 10000보다 큰 숫자 나왔을 때, 50번째로 나오는 숫자 위치와 그 숫자는 무엇인가요?
x=np.where(a > 10000)
a[x[0][49]]

# 500보다 작은 숫자들 중 가장 마지막으로 나오는 숫자 위치와 그 숫자는 무엇인가요?
y = a[np.where(a<500)] # method 1
y[-1]

x = np.where(a<500) # method 2
a[x[0][-1]]

# 벡터 함수 사용
# np.sum(), np.mean(), mp.median(), np.std()

# 빈 칸을 나타내는 방법
# 벡터 안에 nan이 들어있는 경우, 계산 값 nan
a = np.array([20, np.nan, 13, 24, 309])
a + 3
np.mean(a)
np.nan +3
np.nanmean(a) # nan 무시 옵션
np.nan_to_num(a,nan=0)

False
a = None
# 아무런 값도 없는 상태, 특수한 상수, Nonetype
b = np.nan
# 수치 연산에서 정의되지 않은 값이나 잘못된 값, float
# 데이터 분석에서 결측값을 나타내기 위해 사용
# np.nan과의 비교는 np.isnan() 사용
b
a
b + 1
a + 1

# 빈 칸을 제거하는 방법 (nan이 생략된 벡터 얻을 수 있음)
# np.isnan()함수는 벡터의 원소가 nan인지 아닌지 알려줌
# nan인 경우 Ture, 그렇지 않을 경우 Faslse
~np.isnan(a)
a_filtered = a[~np.isnan(a)]
a_filtered

value = np.nan
if np.isnan(value):
print("값이 NaN입니다.")

# 벡터 합치기
str_vec = np.array(["사과", "배", "수박", "참외"])
str_vec
str_vec[[0, 2]]

mix_vec = np.array(["사과", 12, "수박", "참외"], dtype=str)
mix_vec

# np.concatenate()
combined_vec = np.concatenate((str_vec, mix_vec))
combined_vec

# np.column_stack(): 벡터들을 세로로 붙여줌
col_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 16)))
col_stacked

# np.vstack(): 벡터들을 가로로 쌓아줌
col_stacked = np.vstack((np.arange(1, 5), np.arange(12, 16)))
col_stacked

# np.resize(): 길이를 강제로 맞춰주고 값을 앞에서부터 채워줌
vec1 = np.arange(1,5)
vec2 = np.arange(12, 18)
vec1 = np.resize(vec1, len(vec2))
vec1

uneven_stacked = np.column_stack((vec1, vec2))
uneven_stacked
uneven_stacked = np.vstack((vec1, vec2))

# 연습문제 1
a = np.array([1,2,3,4,5])
a
a_new = a + 5
a_new

# 연습문제 2
a = np.array([12, 21, 35, 45, 5])
a[0::2]

# 연습문제 3: 주어진 벡터에서 최대값을 찾기
a = np.array([1, 22, 93, 64, 54])
a.max()

# 연습문제 4: 중복된 값을 제거한 새로운 벡터를 생성
a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
np.unique(a)

# 연습문제 5: 주어진 두 벡터의 요소를 번갈아 가면서 합쳐서 새로운 벡터를 생성 [21, 24, 31, ..]
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])
c = np.empty(a.size + b.size, dtype=a.dtype)
c[0::2] = a
c[1::2] = b
c

# 연습문제 6: 다음 a 벡터의 마지막 값은 제외한 두 벡터 a와 b 더한 결과
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9])
c = a[:-1] + b # [:-1]은 리스트의 처음부터 마지막 -1까지 선택 
c

# 홀수
x[[0,2,4]] = a # method 1
x[0::2] = a # method 2

# 짝수
x[[1, 3, 5]] = b # method 1
x [1::2] # method 2
