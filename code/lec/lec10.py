def g(x=3):
    result = x+1
    return result

g()

print(g) # 함수 객체 정보 출력

# 함수 내용 확인
import inspect
print(inspect.getsource(g))

# if .. else 정식
x = 3
if x > 4:
    y = 1
else:
    y = 2
print(y)

# if else 축약
y = 1 if x > 4 else 2

# 리스트 컴프리헨션
# 리스트 x의 각 원소가 0보다 큰지 여부를 확인하고
# 조건에 따라 “양수” 또는 “음수"를 반환
x = [1, -2, 3, -4, 5]
result = ["양수" if value > 0 else "음수" for value in x]
print(result)

# 조건 3개 이상의 경우 elif()
x = 0
if x > 0:
    result = "양수"
elif x == 0:
    result = "0"
else:
    result = "음수"
print(result)

# numpy.select(): 여러 개의 조건을 처리 -> 주어진 조건들에 따라 선택된 값을 반환
import numpy as np
x = np.array([1, -2, 3, -4, 0])
conditions = [x > 0, x == 0, x < 0]
choices = ["양수", "0", "음수"]
result = np.select(conditions, choices, x)
print(result)
# conditions 리스트는 각 조건을 포함
# choices 리스트는 각 조건에 해당하는 결과를 포함
# numpy.select 함수는 각 조건을 평가하여 참인 첫 번째 조건에 해당하는 결과를 반환

# for 반복문(loop) 구문
# for 변수 in 범위:
#     반복 실행할 코드

for i in range(1, 4):
    print(f"Here is {i}")

# for loop list comprehension
print([f"Here is {i}" for i in range(1, 4)])

# zip(): 
names = ["John", "Alice"]
ages = np.array([25, 30])

# zip() 함수로 names와 ages를 병렬적으로 묶음
zipped = zip(names, ages)

# 각 튜플을 출력
for name, age in zipped:
    print(f"Name: {name}, Age: {age}")

# while 문
i = 0
while i <= 10:
    i += 3
    print(i)

# while, break 문
i = 0
while True:
    i += 3
    if i > 10:
        break
    print(i)

import pandas as pd
df=pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})
df
df.apply(max, axis=0) # 열
df.apply(max, axis=1) # 행

def my_func(x, const=3):
    return max(x)**2 + const

my_func([3, 4, 10], 5)

df.apply(my_func, axis=1)

df.apply(my_func, axis=1, const=5)

# apply 함수 적용
import numpy as np
array_2d = np.arange(1,13).reshape((3, 4), order='F')
np.apply_along_axis(max, axis=0, arr=array_2d)

# 함수 환경
y=2

def my_func(x):
    global y

    def my_f(k):
        return k**2
    
    y = my_f(x) + 1
    result = x + y
    return result

my_func(3)
print(y)

def add_many(*args):
    result=0
    for i in args:
        result = result + i # *args에 입력받은 모든 값을 더함
        return result
    
add_many(1, 2, 3)

def first_many(*args):
    return args[0]

first_many(1,2,3)
first_many(4,1,2,3)

def add_mul(choice, *args):
    if choice == 'add':
        result = 0
        for i in args:
            result = result + i
    elif choice == 'mul':
        result = 1
        for i in input:
            result = result * i
    return result

add_mul('add', 5, 4, 3,1)

# 별포 두 개 (**)는 입력값을 딕셔너리로 만들어줌
def my_twostars(choice, **kwargs):
    if choice == 'first':
        return print(kwargs['age'])
    elif choice == 'second':
        return print(kwargs['name'])
    else:
        return print(kwargs)
my_twostars('first', age=24, name='Jinii')
my_twostars('second', age=24, name='Jinii')
my_twostars('all', age=24, name='Jinii')

dict_a={'age': 24, 'name': 'Jinii'}
dict_a['age']