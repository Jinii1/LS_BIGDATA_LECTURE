# 데이터 타입
x = 15
print(x, "는 ", type(x), "형식입니다.", sep='')
y = 3.14159
print(y, "는 ", type(y), "형식입니다.", sep='')

# 문자형 데이터 예제
a = "Hello, world!"
b = 'python programming'

# 여러 줄 문자열
ml_str = """This is
a multi-line
string"""
print(a, type(a))
print(b, type(b))
print(ml_str, type(ml_str))

# 문자열 결합
greeting = '안녕' + ' ' + '파이썬!'
print('결합 된 문자열:', greeting)

# 문자열 반복
laugh = "하" * 3
print("반복 문자열:", laugh)

# 리스트
fruit = ['apple', 'banana', 'cherry']
type(fruit)

numbers = [1, 2, 3, 4, 5]
type(numbers)

mixed_list = [1, 'Hello!', [1, 2, 3]]
type(mixed_list)

# 튜플 생성 예제
a = (10, 20, 30, 40, 50) # a = 10, 20, 30 과 동일
a[0]
a[1]

print("첫번째 좌표:", a[0])
print("마지막 두개 좌표:", a[1:])
a[3:] #해당 인덱스 이상 ##왼쪽:은 이상 오른쪽:은 미만
a[:3] #해당 인덱스 미만
a[1:3]
a[::]

b_int = 42 # datatype: int 
b_int
b_tp = (42,) #datatype: tuple
b_tp
type(b_int)
type(b_tp)

# 리스트와 튜플의 차이
a_list = [10, 20, 30, 40, 50]
a_tp = (10, 20, 30)
a_list[1] = 25
a_tp[1] = 25
a_list[1]
a_tp[1]
a_tp(1)
a[2:4]
a[1:4] #20,30,40만
a[0:3] #10,20,30만
a[2:] #30,40,50만

# 사용자 정의함수
def min_max(numbers):
  return min(numbers), max(numbers)
a = [1, 2, 3, 4, 5]
result = min_max(a)
result
type(a)
print("Minimum and maximum:", result)

# 딕셔너리 생성 예제
person = {
'name': 'John',
'age': 30,
'city': 'New York'
}
print("Person:", person)

yj = {
  'name': 'Yeonjin',
  'age': (22, 20),
  'city': ('한국','벨기에') 
  }
  print("Person:", yj)
  
yj.get('age')[0]
yj.get('age')[:1]
yj_age = yj.get('age')
yj_age = yj.get('age')
yj_age
yj_age[0]

# 집합
fruits = {'apple', 'banana', 'cherry', 'apple'}
print("Fruits set:", fruits)
fruits
type(fruits)

# 빈 집합 생성
empty_set = set ()
empty_set

empty_set.add(1)
empty_set.add('apple')
empty_set.add('apple')
empty_set
empty_set.add('banana')
empty_set.add('apple')
empty_set
empty_set.remove('banana')

# 집합 간 연산
other_fruits = {'berry', 'cherry'}
union_fruits = fruits.union(other_fruits) #합집합
intersection_fruits = fruits.intersection(other_fruits) #교집합
print("Union of fruits:", union_fruits)
print("Intersection of fruits:", intersection_fruits)
union_fruits

# 논리형 데이터 예제
p = True
q = False
print(p, type(p))
print(q, type(q))
print(p + p) # True는 1로, False는 0으로 계산됩니다.

age = 10
is_active = True
is_greater = age > 5 # True 반환
is_equal = (10 == 5) # False 반환
print("Is active:", is_active)
print("Is 10 greater than 5?:", is_greater)
print("Is 10 equal to 5?:", is_equal)

#조건문
a=3
if (a == 2): ## ==비교연산자
  print("a는 2와 같습니다.")
else:
  print("a는 2와 같지 않습니다.")

# 숫자형을 문자열형으로 변환
num = 123
str_num = str(num)
print("문자열:", str_num, type(str_num))

# 문자열형을 숫자형(실수)으로 변환
num_again = float(str_num)
print("숫자형:", num_again, type(num_again))

# 리스트와 튜플 변환

set_example = {'a', 'b', 'c'}
# 딕셔너리로 변환 시, 일반적으로 집합 요소를 키 또는 값으로 사용
dict_from_set = {key: True for key in set_example}
print("Dictionary from set:", dict_from_set)

# 교재 63페이지
# ! pip install seaborn

import seaborn
import matplotlib.pyplot as plt

var = ['a', 'a', 'b', 'c']
var

seaborn.countplot(x=var)
plt.show()

import seaborn as sns
sns.countplot(x=var)
plt.show()

plt.clf() ## 이전 그래프 설정 지우고 다시 시작하려고

df = sns.load_dataset('titanic')
sns.countplot (data = df, x= 'sex')
sns.countplot (data = df, x= 'sex', hue = 'sex') ## hue: 바 별로 색깔 다르게
plt.show()

?sns.countplot ## 해당 함수에 대한 설명 알려줌

sns.countplot(data = df, x= 'class')
sns.countplot(data = df, x= 'class', hue = 'alive')
sns.countplot(data=df,
y='class',
hue='alive',
orient="v")  ## orient 
plt.show()

! pip install scikit-learn
import sklearn.metrics as met
from sklearn import metrics

import pandas as pd
exam = pd.read_csv('C:/Users/USER/Downloads/exam.csv')
exam

exam.head()
exam.head(10)
exam.tail()
exam.shape
exam.info()
exam.describe()


import os
print(os.getcwd())
import pandas as pd
df_exam = pd.read_excel('excel_exam.xlsx')
df_exam

import os
print(os.getcwd())
import pandas as pd
df_exam = pd.read_csv('exam.csv')
df_exam
