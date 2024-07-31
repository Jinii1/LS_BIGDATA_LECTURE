fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "apple", 3.5, True]

# 리스트의 특징
# 1. 순서 o (각 원소는 인덱스를 통해 접근 o)
# 2. 변경 가능
# 3. 중복 허용 (동일값 여러번)
# 4. 다른 데이터 타입 허용 (숫자형문자형논리형)

# 빈 리스트 생성
empty_list1 = [] # 대괄호 사용
empty_list2 = list() # list함수 사용

# 초기값을 가진 리스트 생성
numbers = [1, 2, 3, 4, 5]
range_list = list(range(5)) # range 함수로 생성한 리스트
range_list

range_list[3] = 'LS 빅데이터 스쿨'

# 두번째 원소에 ['1st', '2nd', '3rd'] 정보 넣기
range_list[1] = ['1st', '2nd', '3rd']
range_list

# '3rd'만 가져오고 싶다면?
range_list[1][2]

# 리스트 내포 Comprehension: 조건이나 반복을 사용하여 리스트를 생성
# 표현식 for 항목 in 반복 가능한 객체
# 1. 대괄호로 쌓여져있다 -> 리스트다.
# 2. 넣고 싶은 수식표현을 x를 사용해서 표현
# 3. for .. in ..을 사용해서 원소정보 제공
list(range(10))
squares = [x**2 for x in range(10)]
squares

# 3, 5, 2, 15의 3제곱
my_squares = [x**3 for x in [3, 5, 2, 15]]
my_squares

# Numpy 배열이 와도 가능
import numpy as np
np.array([3, 5, 2, 15])
my_squares = [x**3 for x in np.array([3, 5, 2, 15])]
my_squares

# Pandas 시리즈 와도 가능
import pandas as pd
exam = pd.read_csv('data/exam.csv')
my_squares = [x**3 for x in exam['math']]
my_squares

# 리스트 합치기
3 + 2
'안녕' + '하세요'
'안녕' * 3

# 리스트 연결
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined_list = list1 + list2

(list1 * 3) + (list2 * 5)

# 리스트 각 원소별 반복
# for x in numbers는 numbers 리스트의 각 항목을 반복
# 내부의 for _ in range(4) 루프는 각 항목에 대해 4번 반복
numbers = [5, 2, 3]
repeated_list = [x for x in numbers for _ in range(4)]
repeated_list = [x for x in numbers for _ in range(4)]
repeated_list = [x for x in numbers for y in range(4)]
repeated_list

# for 루프 문법
# for i in 범위:
#    작동방식
for x in [4, 1, 2, 3]:
    print(x)
    
for i in range(5):
    print(i**2)

# 리스트를 하나 만들고 for 루프를 사용해서 2, 4, 6 .., 20의 수를 채워 넣자
mylist=list(range(1, 11))
[i for i in range (2,21,2)]

mylist = []
for i in range(1, 11):
    mylist.append(i*2) # append: 주어진 맨 마지막 자리에 숫자 추가
mylist

mylist = [0] * 10
for i in range (10):
    mylist[i] = 2 * (i + 1)
mylist

# 인덱스 공유해서 카피하기
mylist_b = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
mylist = [0] * 10

for i in range(10):
    mylist[i] = 2 * (i +1)
mylist

for i in range(10):
    mylist[i] = mylist_b[i]
mylist[i]

# 퀴즈: mylist_b의 홀수번째 위치에 있는 숫자들만 mylist에 가져오기
mylist_b = [2, 4, 6, 80, 10, 12, 24, 35, 23, 20]
mylist = [0] * 5

for i in range(5):
    mylist[i] = mylist_b[i * 2]
mylist

# 리스트 컴프리헨션으로 바꾸는 방법
# 바깥은 무조건 대괄호로 묶어줌: 리스트 반환하기 위해서
# for 루프의 :는 생략
# 실행부분을 먼저 써준다
# 결과값을 발생하는 표현만 남겨두기
mylist = []
[mylist.append(i*2) for i in range(1, 11)] # <- 얘를 바꾸면

[i * 2 for i in range (1, 11)] # 1부터 10까지 숫자 두배로 만들어 리스트 반환
# for i in range(1, 11): i*2와 같은 의미
[x for x in numbers] # numbers=[5, 2, 3] 리스트 요소 그대로 반환

for i in [0, 1, 2]: # ( = range(3))
    for j in [0, 1]: # (= range(2))
         print(i, j)
# 위 식을 리스트 컴프리헨션 변환
[(i, j) for i in [0, 1, 2] for j in [0, 1]]


for i in [0, 1]:
    for j in [4, 5, 6]:
         print(i, j)
# 위 식을 리스트 컴프리헨션 변환
[(i, j) for i in [0, 1] for j in [4, 5, 6]]

# numbers 리스트의 각 요소를 4번씩 출력
numbers = [5, 2, 3]
for i in numbers:
    for j in range(4): # j의 range 뒤에 있는 건 그냥 반복만
        print(i)
# 위 식을 리스트 컴프리헨션 변환
[i for i in numbers for j in range(4)]

repeated_list = [x for x in numbers for _ in range(4)]

# _의 의미
# 1. 앞에 나온 값을 가리킴
5 + 4
_ + 6 # _는 9를 의미

# 값 생략, 자리차지 (placeholder)
a, _, b = (1, 2, 4)
a; b
# _
# _ = None
# del _

# 원소 체크
fruits = ["apple", "banana", "cherry"]
fruits
'banana' in fruits

# 각 원소별로 체크하고 싶은 경우
# [x == "banana" for x in fruits]
mylist=[]
for x in fruits:
    mylist.append(x == 'banana')
mylist

# 바나나의 위치를 뱉어내게 하려면?
fruits = ['apple', 'apple', 'banana', 'cherry']
import numpy as np
fruits = np.array(fruits)
int(np.where(fruits == 'banana')[0][0])

# 원소 거꾸로 써주는 reverse
fruits = ["apple", "banana", "cherry"]
fruits.reverse()

# 리스트 끝에 원소 추가
fruits.append('pineapple')

# 원소 삽입
fruits.insert(2, 'grape')

# 원소 제거
fruits.remove('grape')

import numpy as np

# 넘파이 배열 생성
fruits = np.array(["apple", "banana", "cherry", "apple", "pineapple"])

# 제거할 항목 리스트
items_to_remove = np.array(["banana", "apple"])

# 마스크(논리형 벡터) 생성
mask = ~np.isin(fruits, items_to_remove)
np.isin(fruits, items_to_remove)

# 불리언 마스크를 사용하여 항목 제거
filtered_fruits = fruits[mask]
print("remove() 후 배열:", filtered_fruits)
