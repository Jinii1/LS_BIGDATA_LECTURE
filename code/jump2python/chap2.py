## 02-1 숫자형
10 * 18 ** 2 + 2*11
14 % 3 # 14를 3으로 나눈 나머지
14 // 3 # 14를 3으로 나눈 몫
# -----------------------------------------------------------------------

## 02-2 문자열 자료형
# 여러 줄인 문자열을 변수에 대입하고 싶을 때
multiline = ''' Life is too short
You need python
'''
print(multiline)

multiline = 'Life is too short\nYou need python'

print('-' * 50)
print('My program')
print('-' * 50)

a = 'You need python'
len(a)

# 문자열 인덱싱과 슬라이싱
# 인덱싱: 특정 값을 뽑아 낸다
# 슬라이싱: 잘라낸다
a = 'Life is too short, You need Python'
a[4]

a[0:4] # 0<=a<4
# 슬라이싱 기법으로 a[시작 번호:끝 번호]를 지정할 때 끝 번호에 해당하는 문자는 포함 x

a = 'Pithon' # 을 python으로 바꾸기
a = a[:1] + 'y' + a[2:]

# 문자열 포매팅
# 문자열 안에 어떤 값을 삽입하는 경우
'I eat %d apples' %3
'I eat %s apples' %'three'
number=10
day='three'
'I eat %d apples. So I was sick for %s days' %(number, day)

'%10s' % 'hi'
'%-10s' % 'hi'

# format 함수를 사용한 포매팅
'I eat {0} apples'.format('five')
'I ate {number} apples. So I was sick for {day} days'.format(number=10, day=3)

'{0:<10}'.format('hi') # 치환되는 문자열을 왼쪽으로 정렬하고 문자열의 총 자릿수를 10으로
'{0:>10}'.format('hi')
'{0:^10}'.format('hi')
'{0:=^10}'.format('hi')

# f 문자열 포매팅
name = '홍길동'
age = 30
f'나의 이름은 {name}입니다. 나이는 {age}입니다.'

d={'name': '홍길동', 'age': 30}
f'나의 이름은 {d['name']}입니다. 나이는 {d['age']}입니다.'

# format 함수 또는 f 문자열 포매팅을 사용해 !!!python!!! 문자열을 출력해보자
'{0:!^12}'.format('python')
f'{"python":!^12}'

# 문자 개수 세기
a = 'hobby'
a.count('b')

# 위치 알려주기 find
a = 'Python is the best choice'
a.find('b') # 문자열에서 b가 처음 나온 위치
a.find('k')# 찾는 문자나 문자열이 존재하지 않는 경우 -1 반환

# 위치 알려주기 index
a = 'Life is too short'
a.index('t')

# 문자열 삽입 join
','.join('abcd') # abcd 문자열의 각각의 문자 사이에 ',' 삽입
','.join(['a', 'b', 'c', 'd'])

# 소문자 대문자 upper lower
a= 'hi'
a.upper()
b='HI'
b.lower()

# 공백 지우기 strip, lstrip 왼쪽, rstrip 오른쪽
a = '  hi'
a.lstrip()

# 문자열 바꾸기 replace
a = 'Life is too short'
a.replace('Life', 'Your leg') # replace(바뀔 문자열, 바꿀 문자열)

# 문자열 나누기 split
a = 'Life is too short'
a.split()
b='a:b:c:d'
b.split(':')
# -----------------------------------------------------------------------

## 02-3 리스트 자료형
odd = [1, 3, 5, 7, 9]
# 리스트는 대괄호로 감싸 주고 각 요솟값은 쉼표로 구분
a = []
b= [1, 2, 3]
c=['Life', 'is', 'too', 'short']

# 리스트의 인덱싱
a = [1, 2, 3]
a
a[0]

a = [1, 2, 3, ['a', 'b', 'c']]
a[-1]
a[-1][0] # 리스트 ['a', 'b', 'c']에서 'a'값을 인덱싱을 사용해서 끄집어 낸

a = [1, 2, ['a', 'b', ['Life', 'is']]]
# life 문자열 꺼내려면
a[2][2][0]

# 리스트의 슬라이싱
a = [1,2,3,4,5]
a[0:2]

a='12345'
a[0:2]

# a=[1,2,3,4,5] 리스트에서 슬라이싱 기법을 사용하여 리스트 [2,3] 만들기
a=[1,2,3,4,5]
a[1:3]

a=[1,2,3,['a','b','c'], 4,5]
a[2:5]

# 리스트 연산하기
a=[1,2,3]
b=[4,5,6]
a+b
a*3
len(a)
str(a[2]) + 'hi'

# 리스트 수정과 삭제
a=[1,2,3]
a[2]=4
a

del a[1]
a

a=[1,2,3,4,5]
del a[:3]
a

# 리스트 관련 함수
# 리스트에 요소 추가하기 append
a = [1,2,3]
a.append(4)
a.append([5,6])
a

# 리스트 정렬 sort
a = [1,4,3,2]
a.sort()
a

# 리스트 뒤집기 reverse
# 현재 리스트를 그대로 거꾸로 뒤집기
a=['a', 'c', 'b']
a.reverse()
a

# 인덱스 반환 index
# 인덱스값 (위치값) 반환
a = [1,2,3]
a.index(3)

# 리스트 요소 삽입 insert
# a번째 위치에 b를 삽입
a=[1,2,3]
a.insert(0, 4)
a.insert(3, 5)
a

# 리스트 요소 제거 remove
# 첫번째로 나오는 x를 삭제
a = [1,2,3,1,2,3]
a.remove(3)
a

# 리스트 요소 끄집어 내기 pop
# 리스트의 맨 마지막 요소를 리턴하고 그 요소를 삭제
a = [1,2,3]
a.pop()

# 리스트에 포함된 요소 x의 개수 세기 count
# 리스트 안에 x가 몇 개 있는지 조사하여 그 개수를 리턴하는 함수
a=[1,2,3,1]
a.count(1)

# 리스트 확장 extend
# x에는 리스트만 올 수 있으며 원래의 a 리스트에 x리스트를 더함
a=[1,2,3]
a.extend([4,5])
a
# -----------------------------------------------------------------------

## 02-4 튜플 자료형
# 리스트는 [], 튜플은 ()
# 리스트는 요소값의 생성, 삭제, 수정이 가능하지만, 튜플은 요솟값을 바꿀 수 x

t1=()
t2=(1,) # 1개의 요소만을 가질 때 요소 뒤에 쉼표를 반드시 붙여야 함
t3=(1,2,3)
t4=1,2,3 # 소괄호 생략 가능
t5=('a', 'b', ('ab', 'cd'))

# 튜플과 리스트의 가장 큰 차이
# 1. 리스트의 요솟값은 변화 가능, 튜플의 요솟값은 변화 불가능
# 수시로 요솟값을 변화 시켜야 한다면 리스트, 항상 변하지 않길 바란다면 튜플

t1=(1,2,'a','b')
t1
t1[0]
t1[1:]
t2=(3,4)
t3=t1+t2
t4=t2*2

# (1,2,3)이라는 튜플에 값 4를 추가하여 (1,2,3,4)라는 새로운 튜플을 출력
t1=(1,2,3)
t2=(4,)
t1+t2
# -----------------------------------------------------------------------

## 02-5 딕셔너리 자료형
# key를 통해 value를 얻는
dic = {'name': 'Jinii', 'phone': '010-3050-7651', 'birth': '0904'}
a = {'a': [1,2,3]}

a = {1: 'a'}
a[2] = 'b'
a

del a[1]

# ex. 4명의 사람의 각자 특기를 표현할 수 있는 가장 좋은 방법

grade={'pey': 10, 'julliet': 99}
grade['julliet'] # 딕셔너리변수이름[Key]

# key는 고유한 값으로 중복x, 변하지 않는 값을 사용 (tuple 가능, list 불가능)

dic = {'name': 'Jinii', 'phone': '010-3050-7651', 'birth': '0904'}
dic.keys()

dic['name']
dic.get('name')

for k in dic.keys():
            print(k)

# key, value 쌍 얻기 - items
dic.items()
# key, value 쌍 모두 지우기 - clear
dic.clear()

'name' in dic

dic = {'name':'홍길동', 'birth': 1128, 'age': 30}
# -----------------------------------------------------------------------

## 02-6 집합 자료형