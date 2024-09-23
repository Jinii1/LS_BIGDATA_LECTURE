import pandas as pd
import numpy as np

train = pd.read_csv('../data/bike/train.csv')
test = pd.read_csv('../data/bike/test.csv')
submission = pd.read_csv('../data/bike/sampleSubmission.csv')

### 6.3.2 데이터 둘러보기
train.shape
test.shape

train.head()
test.head()

train.columns
test.columns # casual, registered라는 features 빠짐
# 그래서 train data에서도 뺴야 됨

submission.head()

train.info() # train.isna().sum()
test.info()

### 6.3.3 더 효과적인 분석을 위한 피처 엔지니어링
train['datetime'][100] # datetime 100번째 원소
train['datetime'][100].split() # 공백 기준으로 문자열 나누기
train['datetime'][100].split()[0] # 날짜
train['datetime'][100].split()[1] # 시간

## date, year, month, dat, hour, minute, second, weekday 피처 추가
# 날짜 문자열을 다시 연도, 월, 일로 나눠보기
train['datetime'][100].split()[0].split('-')
train['datetime'][100].split()[0].split('-')[0] # 연도
train['datetime'][100].split()[0].split('-')[1] # 월
train['datetime'][100].split()[0].split('-')[2] # 일

# 시간 문자열을 시, 분, 초로 나눠보기
train['datetime'][100].split()[1]
train['datetime'][100].split()[1].split(':')
train['datetime'][100].split()[1].split(':')[0] # 시간
train['datetime'][100].split()[1].split(':')[1] # 분
train['datetime'][100].split()[1].split(':')[2] # 초

# pandas apply() 함수로 앞의 로직을 datetime에 적용해
# 날짜, 연도, 월, 일, 시, 분, 초 피처를 생성
train['date']=train['datetime'].apply(lambda x: x.split()[0])

train['year'] = train['datetime'].apply(lambda x: x.split()[0].split('-')[0])
train['month'] = train['datetime'].apply(lambda x: x.split()[0].split('-')[1])
train['day'] = train['datetime'].apply(lambda x: x.split()[0].split('-')[2])
train['hour'] = train['datetime'].apply(lambda x : x.split()[1].split(':')[0])
train['minute'] = train['datetime'].apply(lambda x : x.split()[1].split(':')[1])
train['second'] = train['datetime'].apply(lambda x : x.split()[1].split(':')[2])

# 요일 피처 생성
from datetime import datetime
import calendar

train['date'][100]
# datetime 타입으로 변경
datetime.strptime(train['date'][100], '%Y-%m-%d')
# 정수로 요일 반환
datetime.strptime(train['date'][100], '%Y-%M-%d').weekday()
# 문자열로 요일 반환
calendar.day_name[datetime.strptime(train['date'][100], '%Y-%M-%d').weekday()]
# apply() 함수로 적용해 요일(weekday) 피처를 추가
train['weekday'] = train['date'].apply(
            lambda dateString:
            calendar.day_name[datetime.strptime(dateString, '%Y-%m-%d').weekday()])

## season, weather 피처는 숫자에서 문자로 변경
# 1,2,3,4 숫자 범주형 데이터로 표현되어 있어 시각화를 위해 문자열로 바꾸기
train['season'] = train['season'].map({1: 'spring',
                                       2: 'summer',
                                       3: 'autumn',
                                       4: 'winter'})
train['weather'] = train['weather'].map({1: 'Clear',
                                       2: 'Mist, Few Clouds',
                                       3: 'Light Snow, Rain, Thunderstorm',
                                       4: 'Heavy Rain, Thunderstorm, Snow, Fog'})

train.head()

### 6.3.4 데이터 시각화
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# 분포도: 수치형 데이터의 집계 값(개수, 비율)을 나타내는 그래프
# 타깃인 count의 분포를 그려보기
# -> 타깃값의 분포를 알면 훈련 시 타깃값을 그대로 사용할지 변환해 사용할지 파악 가능
mpl.rc('font', size=15)
sns.displot(train['count'])
plt.show()
# 타깃값인 count가 0근처에 몰려 있다 = 분포가 왼쪽으로 많이 편향되어 있다
# 회귀 모델이 좋은 성능을 내려면 데이터가 정규분포를 따라야 하는데 그렇지 않기에
# 로그 변환을 함 (count분포와 같이 왼쪽으로 편향되어 있을 때 주로 사용)
sns.displot(np.log(train['count']))
plt.show()
# 피처를 바로 사용해 count를 예측하는 것보다 log(count)예측하는 편이 더 정확
# 다만, 마지막에 지수변환을 하여 실제 타깃값인 count로 복원해야 함

# 연도, 월, 일, 시, 분, 초별로 총 여섯 가지의 평균 대여 수량을 막대그래프로 그리기
# 1. 6개의 그래프 (서브플롯)를 품는 3행 2열짜리 Figure 준비
mpl.rc('font', size=14) # 폰트 크기 설정
mpl.rc('axes', titlesize=15) # 각 축의 제목 크기 설정
figure, axes =plt.subplots(nrows=3, ncols=2) # 3행 2열 Figure 생성
plt.tight_layout() # 그래프 사이에 여백 확보
figure.set_size_inches(10,9) # 전체 Figure 크기를 10x9인치로 설정

# 2. 각 축에 서브플롯 할당
sns.barplot(x='year', y='count', data=train, ax=axes[0,0])
sns.barplot(x='month', y='count', data=train, ax=axes[0,1])
sns.barplot(x='day', y='count', data=train, ax=axes[1,0])
sns.barplot(x='hour', y='count', data=train, ax=axes[1,1])
sns.barplot(x='minute', y='count', data=train, ax=axes[2,0])
sns.barplot(x='second', y='count', data=train, ax=axes[2,1])

# 3. 세부 설정
axes[1, 0].tick_params(axis='x', labelrotation=90)
axes[1, 1].tick_params(axis='x', labelrotation=90)
plt.show()