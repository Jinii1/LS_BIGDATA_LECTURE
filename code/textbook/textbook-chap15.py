# 머신러닝 모델 = 함수 (컴퓨터가 데이터에서 패턴을 찾아 스스로 규칙을 정함)
# 예측변수와 타겟변수 (예측하고자 하는 변수가 출력하는 값)
# 미래의 값을 예측하는 용도로 자주 사용

# 의사결정나무 모델
# 1. 타겟 변수를 가장 잘 분리하는 예측 변수 선택
# 답변의 비율 차이가 크면 클수록 예측 변수가 타겟 변수를 잘 분리함 (범주형)
# 모든 경우의 수대로 질문 후보를 만들어 비교 (연속형)
# 2. 첫 번째 질문의 답변에 따라 데이터를 두 노드로 분할
# 질문의 답변이 같아서 함께 분류된 집단을 노드라고 부름
# 3. 각 노드에서 타겟 변수를 가장 잘 분리하는 예측 변수 선택
# 4. 노드가 완벽하게 분리될 때까지 반복

# 특징
# 1. 노드마다 분할 횟수가 다르다
# 2. 노드마다 선택되는 예측 변수가 다르다
# 3. 어떤 예측 변수는 모델에서 탈락한다

# 15-2
# adult 데이터를 이용해 인적 정보로 소득을 예측하는 의사결정나무 모델 만들기
import pandas as pd
import numpy as np
import os
cwd=os.getcwd()

df=pd.read_csv('../../data/adult.csv')
# income을 타겟 변수, 나머지 13개를 예측 변수로 사용

# 타겟 변수 전처리
df['income'].value_counts(normalize=True)
df['income']= np.where(df['income']=='>50K', 'high', 'low')
df=df.drop(columns='fnlwgt')
df.info()

# 문자 타입 변수를 숫자 타입으로 바꾸기
# 원핫 인코딩 (변수의 범주가 특정 값이면 1, 그렇지 않으면 0으로 변환)
df_tmp=df[['sex']]
df_tmp.info()
df_tmp.value_counts()
df_tmp=pd.get_dummies(df_tmp) # sex가 female이면 true
df_tmp

target=df['income'] # income 추출
df=df.drop(columns='income')
df=pd.get_dummies(df)