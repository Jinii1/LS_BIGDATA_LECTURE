import pandas as pd
import numpy as np

train = pd.read_csv('../../data/ml/chap6/train.csv')
test = pd.read_csv('../../data/ml/chap6/test.csv')
submission = pd.read_csv('../../data/ml/chap6/submission.csv')

### 6.3.2 데이터 둘러보기
train.shape
test.shape

train.head()
test.head()

train.columns
test.columns
# test data에서 casual, registered라는 features 빠짐
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

train['year'] = train['datetime'].apply(lambda x: x.split()[0].split('-')[0]) # 연
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
print(datetime.strptime(train['date'][100], '%Y-%m-%d'))
# 정수로 요일 반환
print(datetime.strptime(train['date'][100], '%Y-%m-%d').weekday())
# 문자열로 요일 반환 (그래프 보기 쉬우려고)
print(calendar.day_name[datetime.strptime(train['date'][100], '%Y-%M-%d').weekday()])

# apply() 함수로 적용해 요일(weekday) 피처를 추가
train['weekday'] = train['date'].apply(
            lambda dateString:
            calendar.day_name[datetime.strptime(dateString, '%Y-%m-%d').weekday()])

## season, weather 피처는 숫자에서 문자로 변경
# 1,2,3,4 숫자 범주형 데이터로 표현되어 있어 시각화를 위해 문자열로 바꾸기
# train['season'] = train['season'].map({1: 'spring',
#                                        2: 'summer',
#                                        3: 'autumn',
#                                        4: 'winter'})
# train['weather'] = train['weather'].map({1: 'Clear',
#                                        2: 'Mist, Few Clouds',
#                                        3: 'Light Snow, Rain, Thunderstorm',
#                                        4: 'Heavy Rain, Thunderstorm, Snow, Fog'})

# train.head()



### 6.3.4 데이터 시각화
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


## 분포도: 수치형 데이터의 집계 값(개수, 비율)을 나타내는 그래프
# 타깃인 count의 분포를 그려보기
# -> 분포 파악 후, 훈련 시 타깃값을 그대로 사용할지 변환해 사용할지 파악 가능
mpl.rc('font', size=15)
sns.displot(train['count'])
# 타깃값인 count가 0근처에 몰려 있다 = 분포가 왼쪽으로 많이 편향되어 있다
# 회귀 모델이 좋은 성능을 내려면 데이터가 정규분포를 따라야 하는데 그렇지 않기에
# 로그 변환을 함 (count분포와 같이 왼쪽으로 편향되어 있을 때 주로 사용)
sns.displot(np.log(train['count']))
# 마지막에 지수변환을 하여 실제 타깃값인 count로 복원해야 함


## 연도, 월, 일, 시, 분, 초별로 총 여섯 가지의 평균 대여 수량을 막대그래프로 그리기
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

## 박스플롯: 범주형 데이터에 따른 수치형 데이터 정보를 나타내는 그래프
# 1. m행 n열 Figure 준비
figure, axes = plt.subplots(nrows=2, ncols=2)
plt.tight_layout()
figure.set_size_inches(10, 10)

# 2. 서브플롯 할당 (계절, 날씨, 공휴일, 근무일별 대비 수량 박스플롯)
sns.boxplot(x='season', y='count', data=train, ax=axes[0,0])
sns.boxplot(x='weather', y='count', data=train, ax=axes[0,1])
sns.boxplot(x='holiday', y='count', data=train, ax=axes[1,0])
sns.boxplot(x='workingday', y='count', data=train, ax=axes[1,1])

# 3. 세부 설정
axes[0,0].set(title='Box Plot On Count Across Season')
axes[0,1].set(title='Box Plot On Count Across Weather')
axes[1,0].set(title='Box Plot On Count Across Holiday')
axes[1,1].set(title='Box Plot On Count Across Workingday')
# 봄에 가장 적고, 가을에 가장 많다
# 날씨 좋을 때 많고, 날씨 안 좋을 때 적다
# 0은 공휴일x, 1은 공휴일을 의미, 중앙값은 비슷한데 공휴일이 아닐 때 이상치가 많음
# 근무일일 때 이상치가 많음


## 포인트플롯: 범주형 데이터에 따른 수치형 데이터의 평균과 신뢰구간을 점과 선으로 표시
# workingday, holiday, weekday, season, weather에 따른 hour별 count 그래프 그리기
# 1. m행 n열 Figure 준비
mpl.rc('font', size=11)
figure, axes=plt.subplots(nrows=5)
figure.set_size_inches(12, 18)

# 2. 서브플롯 할당
sns.pointplot(x='hour', y='count', data=train, hue='workingday', ax=axes[0])
sns.pointplot(x='hour', y='count', data=train, hue='holiday', ax=axes[1])
sns.pointplot(x='hour', y='count', data=train, hue='weekday', ax=axes[2])
sns.pointplot(x='hour', y='count', data=train, hue='season', ax=axes[3])
sns.pointplot(x='hour', y='count', data=train, hue='weather', ax=axes[4])
plt.show()


## 회귀선을 포함한 산점도 그래프 (수치형 데이터 간 상관관계 파악 용이)
# 수치형 데이터인 온도, 체감온도, 풍속, 습도별 대여 수량을 그려보기
# 1. m행 n열 Figure 준비
mpl.rc('font', size=15)
figure, axes=plt.subplots(nrows=2, ncols=2)
plt.tight_layout()
figure.set_size_inches(7, 6)

# 2. 
sns.regplot(x='temp', y='count', data=train, ax=axes[0,0],
            scatter_kws={'alpha': 0.2}, line_kws={'color':'blue'}) # 점 투명도, 회귀선 색상
sns.regplot(x='atemp', y='count', data=train, ax=axes[0,1],
            scatter_kws={'alpha': 0.2}, line_kws={'color':'blue'})
sns.regplot(x='windspeed', y='count', data=train, ax=axes[1,0],
            scatter_kws={'alpha': 0.2}, line_kws={'color':'blue'})
sns.regplot(x='humidity', y='count', data=train, ax=axes[1,1],
            scatter_kws={'alpha': 0.2}, line_kws={'color':'blue'})
# 온도와 체감온도가 높을수록 대여 수량 많음
# 습도는 낮을수록 대여를 많이 함
# 풍속이 셀수록 대여 많이 함 (?) -> windspeed 결측치 많음 -> 적절한 처리 필요


## 히트맵: 수치형 데이터끼리 상관관계
corrMat=train[['temp', 'atemp', 'windspeed', 'humidity', 'count']].corr()
fig, ax=plt.subplots()
sns.heatmap(corrMat, annot=True) # 상관관계 숫자 표시
ax.set(title='Heatmap Of Numerical Data')
# 온도와 대여수량 0.39 양의 상관관계
# 습도와 대여수량 -0.32 음의 상관관계 (습도가 낮을수록 대여수량 많다)
# 풍속과 대여수량은 0.1로 상관관계가 매우 약함

### 6.4 베이스라인 모델
# 이상치 제거
train = train[train['weather']!=4]

# train과 test에 같은 피처 엔지니어링을 적용하기 위해 데이터 합치기
all_data=pd.concat([train, test], ignore_index=True) # 원래 인덱스 무시

# 파생 피처 추가 (날짜, 연도, 월, 시, 요일)
all_data['date']= all_data['datetime'].apply(lambda x: x.split()[0])
all_data['year']= all_data['datetime'].apply(lambda x: x.split()[0].split('-')[0])
all_data['month']= all_data['datetime'].apply(lambda x: x.split()[0].split('-')[1])
all_data['hour']= all_data['datetime'].apply(lambda x: x.split()[1].split(':')[0])
all_data['weekday']=all_data['date'].apply(lambda dateString: datetime.strptime(dateString, '%Y-%m-%d').weekday())

# 필요 없는 피처 제거
all_data=all_data.drop(['casual', 'registered', 'datetime', 'date', 'month', 'day', 'minute', 'second', 'windspeed'], axis=1)

# 데이터 나누기
# 타깃값이 있으면 train, 없으면 test
X_train=all_data[~pd.isnull(all_data['count'])]
X_test=all_data[pd.isnull(all_data['count'])]

X_train=X_train.drop(['count'], axis=1)
X_test=X_test.drop(['count'], axis=1)

X_test.info()
X_train.shape

y=train['count']

## 평가지표 계산 함수 작성
def rmsle(y_true, y_pred, convertExp=True):
            # 지수변환
            if convertExp:
                    y_true=np.exp(y_true)
                    y_pred=np.exp(y_pred)

            # 로그변환 후 결측값을 0으로 변환
            # 값이 0일 경우 log 함수 정의가 안 돼서 +1
            # nan값을 0으로 처리
            # np.log1p(y)로 표현 가능
            log_true = np.nan_to_num(np.log(y_true+1))
            log_pred = np.nan_to_num(np.log(y_pred+1))
            
            # RMSLE 계산
            output = np.sqrt(np.mean((log_true - log_pred)**2))
            return output              

### 6.4.3 모델 훈련
from sklearn.linear_model import LinearRegression
Linear_reg_model = LinearRegression()
log_y=np.log(y) # 타깃값 로그변환
Linear_reg_model.fit(X_train, log_y) # 모델 훈련

### 6.4.4 모델 검증
preds=Linear_reg_model.predict(X_train) # RMSLE 값 확인
# 훈련과 예측을 코드로 어떻게 구현하는지 간단히 보여주려는 것으로
# 원래는 훈련시 훈련 데이터 사용, 테스트 시 테스트 데이터 사용해야함

# 제출하기 전까지 테스트 데이터로 RMSLE 구할 수 x (예측값과 실제값이 있어야 하기 때문)
# 이런 경우 보통 훈련 데이터를 훈련용과 검증용으로 나눠서
# 훈련용으로 모델을 훈련하고 검증용으로 훈련된 모델의 성능을 평가

print(f'선형 회귀의 RNSLE 값: {rmsle(log_y, preds, True): 4f}')

linearreg_preds=Linear_reg_model.predict(X_test) # test 파일로 예측

submission['count'] = np.exp(linearreg_preds) # 지수변환
submission.to_csv('../../data/bike/submission.csv', index=False)

### 6.5 릿지 회귀 모델 
# 하이퍼파라미터 최적화 (모델훈련)
# 그리드서치: 하이퍼파라미터를 격자처럼 촘촘하게 순회하며 최적의 하이퍼파라미터 값 찾기
# 각 하이퍼파라미터를 적용한 모델마다 cross-validation하며 성능 측정하여
# 최종적으로 성능이 가장 좋았을 때의 하이퍼파라미터 값을 찾아줌
# 교차 검증 평가점수는 보통 에러 값이기 때문에 낮을 수록 좋음

# 모델 생성
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

ridge_model = Ridge()

# 그리드서치 객체 생성
# 하이퍼파라미터의 값을 바꿔가며 모델의 성능을 교차 검증으로 평가해 최적의 하이퍼파라미터 값을 찾아줌
ridge_params = {
    'alpha': np.logspace(-2, 3, 10),  # 0.01에서 1000까지 로그 스케일로 10개의 값
    'max_iter': [3000]  # max_iter는 한 가지 값으로 고정
}
# 모델 훈련 시 최대로 허용하는 반복 횟수, 모델의 정규화 강도를 조절하는 값
rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)
gridsearch_ridge_model = GridSearchCV(estimator=ridge_model, # 분류 및 회귀 모델
                                      param_grid=ridge_params,# 하이퍼파라미터 값 지정
                                      scoring=rmsle_scorer, # 평가지표
                                      cv=5) # 교차 검증 분할 수
log_y=np.log(y)
gridsearch_ridge_model.fit(X_train, log_y)
print('최적 하이퍼파라미터: ', gridsearch_ridge_model.best_params_)

preds= gridsearch_ridge_model.best_estimator_.predict(X_train)
print(f'릿지 회귀의 RNSLE 값: {rmsle(log_y, preds, True): 4f}')

### 6.6 라쏘 회귀 모델
from sklearn.linear_model import Lasso
lasso_model=Lasso()
lasso_alpha=1/np.array([0.1, 1, 2, 3, 4, 10, 30, 100, 200, 300, 400, 800, 900, 1000])
lasso_params = {'max_iter': [3000], 'alpha': lasso_alpha}

gridsearch_lasso_model = GridSearchCV(estimator=lasso_model,
                                      param_grid=lasso_params,
                                      scoring=rmsle_scorer,
                                      cv=5)

log_y=np.log(y)
gridsearch_lasso_model.fit(X_train, log_y)
print('최적 하이퍼파라미터: ', gridsearch_lasso_model.best_params_)

preds= gridsearch_lasso_model.best_estimator_.predict(X_train)
print(f'릿지 회귀의 RNSLE 값: {rmsle(log_y, preds, True): 4f}')

# alpha와 max_iter 값 설정 시 유용한 꿀팁이 있을까?

# alpha값은 로그 스케일로 설정하는 것이 유리 (정규화 강도가 기하급수적으로 변할 수 있기에)
# 작은 alpha 값에서 적당히 탐색한 후, 필요한 경우 값을 키워가며 규제 강도 조절
# 과적합이 우려될 때는 큰 alpha, 과소적합 가능성이 있으면 작은 alpha 값을 사용
# Lasso는 변수를 제거하므로 작은 alpha가 유리하고, Ridge는 변수를 유지하면서 가중치를 줄이므로 큰 alpha가 더 적합

# max_iter은 기본적으로 1000~3000


### 6.7 랜덤포레스트 회귀 모델
from sklearn.ensemble import RandomForestRegressor
randomforest_model=RandomForestRegressor()
rf_params = {'random_state':[42], 'n_estimators': [100, 120, 140]} # 결정트리갯수
gridsearch_random_forest_model = GridSearchCV(estimator=randomforest_model,
                                              param_grid=rf_params,
                                              scoring=rmsle_scorer,
                                              cv=5)

log_y=np.log(y)
gridsearch_random_forest_model.fit(X_train, log_y)
print('최적 하이퍼파라미터: ', gridsearch_random_forest_model.best_params_)

preds= gridsearch_random_forest_model.best_estimator_.predict(X_train)
print(f'랜덤포레스트 회귀의 RNSLE 값: {rmsle(log_y, preds, True): 4f}')

### 6.7.3 예측 및 결과 제출
rf_preds=gridsearch_random_forest_model.best_estimator_.predict(X_test)
import seaborn as sns
import matplotlib.pyplot as plt

figure, axes = plt.subplots(ncols=2)
figure.set_size_inches(10, 4)
sns.histplot(y, bins=50, ax=axes[0])
sns.histplot(np.exp(rf_preds), bins=50, ax=axes[1])

submission['count']=np.exp(rf_preds)
submission.to_csv('../../data/ml/chap6/submission.csv', index=False)

# 베이스라인 모델 -> 성능 개선 (타깃값 변환, 이상치 제거, 파생 피처 추가, 피처 제거)
# -> 선형 회귀, 릿지, 라쏘, 랜ㄴ더모레스트 회귀, 그리드서치