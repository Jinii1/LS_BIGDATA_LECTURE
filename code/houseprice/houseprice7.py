# Neighborhood 변수를 더미 변수로 불러와서

# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test=pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

# ## 이상치 탐색
# house_train=house_train.query("GrLivArea <= 4500")

## 회귀분석 적합(fit)하기
# house_train["GrLivArea"]   # 판다스 시리즈
# house_train[["GrLivArea"]] # 판다스 프레임
x = house_train[["GrLivArea", "GarageArea"]]
y = house_train["SalePrice"]

# ejal qustnfh qusghks
neighborhood_dummies = pd.get_dummies(house_train['Neighborhood'],
                                  drop_first=True)
x= pd.concat([house_train[["GrLivArea", "GarageArea"]], neighborhood_dummies], axis=1)
y = house_train["SalePrice"]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

neighborhood_dummies_test = pd.get_dummies(house_test['Neighborhood'],
                                  drop_first=True)

test_x=pd.concat([house_test[['GrLivArea', 'GarageArea']],
                        neighborhood_dummies_test], axis=1)

# 결측치 확인
test_x["GrLivArea"].isna().sum()
test_x["GarageArea"].isna().sum()
test_x=test_x.fillna(house_test["GarageArea"].mean())

pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission10.csv", index=False)


특정 neighborhood 범주에 대한 지도가 나와 그거에 대한 회귀직선을 그려 그런 걸 할 수 있다
대쉬보드에는 여러 섹션이 있다
강남구 종로구 뭐 이런 ...
강남구에 해당하는 지역 지도에 점을 찍을 수 있잖아?
이 집에 해당하는 것중에 오른쪽에는 우리가 방금 그린 회귀분석을 그릴 수 있지 않을까?
뭘 그리라는거지
데이터 필터링을 해서 특정 지역을 눌렀을 때 그 지역의 지도랑 groudnare, 대비 price가 나오고
price를 잘 나타내는 회귀직선을 그리고 
탭을 

대쉬보드에 탭을 마을 별로 설정할 수 있으니까 이 동네에 대한 dashboard가 나올 수 있게 만들수 있을것
그럼 그거에 대한 동일 구조가 나올것
