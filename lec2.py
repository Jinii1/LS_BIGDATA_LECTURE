#Ctrl+Enter
#Shift+화살표:블록

a=1
a=

#파워쉘 명령어 리스트
#ls:파일목록
#cd:폴더 이동
#.현재폴더
#.. 상위폴더
#Tab\Shift Tab 자동완성

#Show folder in new window
#해당위치 탐색기

#open your new terminal

#cls: 화면정리

a=10

a='안녕하세요'
a="'안녕하세요!' 라고 아빠가 말했다."
a
a=[1,2,3]
a

var1=[1,2,3]
var2=[4,5,6]
var1+var2

a='안녕하세요'
b='LS 빅데이터 스쿨!'
a+b
a+' '+b
print(a)

num1 = 3
num2 = 5
num1 + num2

a=10
b=3.3

print ("a+b=", a+b)
print ("a-b=", a-b)
print ("a*b=", a*b)
print ("a/b=", a/b)
print ("a%b=", a%b)
print ("a//b=", a//b)
print ("a**b=", a**b)

#Shift + Alt + 아래화살표: 아래로 복사
#Ctrl + Alt + 아래화살표: 커서 여러개
(a ** 2) // 7
(a ** 2) // 7
(a ** 2) // 7
(a ** 2) // 7

a
b

a == b
a != b
a > b
a < b
a<=b
a>=b

a=(2**4+12453//7)%8

b=(9**7/12)*(36452%253)
print(a>b)

a= ((2**4) + (12453//7)) % 8 #2의 4승과 12453을 7로 나눈 몫을 더해서 8로 나눴을 때 나머지
b= (((9 ** 7) / 12) * (36452%253)) #9의 7승을 12로 나누고, 36542를 253로 나눈 나머지에 곱한 수
a
b
a<b

user_age = 25
is_adult = user_age >= 18 ## True or False라는 답 가질 수 있는 변수가 됨
print("성인입니까?", is_adult)

False = 3 ## 예약어라서 다르게 표시되고 사용할 수 x

a = "True"
b= TRUE ## TRUE는 예약어x (예약어는 True), TRUE는 변수가 될 수 있는 앤데 우리가 그런 변수 만든 적 없어서
c = true
d= True

TRUE=2
b=TRUE
b

# True, False
a = True
b = False

a and b
a or b
not a

# True: 1
# False: 0
True + True
True + False
False + False

# and 연산자
True and False
True and True
False and False
False and True

# or 연산자
True or True
True or False
False or True
False or False

# and는 곱셈으로 치환 가능
True * False
True * True
False * False
False * True

a = True
b = False
a or b
min(a + b, 1)

# 복합 대입 연산자
a = 3
a += 10 # a = a + 10

a -= 4
a

a %= 3
a

a += 12
a ** 2
a /= 7
a

str1 = 'Hello'
str1 + str1
str1 * 3

str1 = "Hello! "
# 문자열 반복
repeated_str = str1 * 3
print("Repeated string:", repeated_str)

# 정수: int(eger)
# 실수: float (double)

# 단항 연산자
x = 5
x
+x
-x
~x

# binary
bin(5)
bin(-5) # 얘네 값은 문자 ''표시되어있기 때문, 0b는 이진수를 뜻하는 의미 (파이썬에서)
bin (-6)

max (3, 4)
var1 = [1, 2, 3]
sum(var1)

import pydataset
pydataset.data() # ()로 함수라는 걸 알 수 있음, 입력값이 아니라 어떤 데이터가 있는지

df=pydataset.data('AirPassengers') #데이터의 목록을 반환하는 행동을 함 이름을 넣을 땐 실제로 불러와서 반환
df

import pandas as pd

import pandas as pd
import numpy as np
print(np.__version__)
print(pd.__version__)
