import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

house_train=pd.read_csv("../../data/houseprice/train.csv")
house_test=pd.read_csv("../../data/houseprice/test.csv")
sub_df=pd.read_csv("../../data/houseprice/sample_submission.csv")

## NaN 채우기
# 각 숫치형 변수는 평균 채우기
# 각 범주형 변수는 Unknown 채우기
house_train.isna().sum()
house_test.isna().sum()

## 숫자형 채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)
house_train[qual_selected].isna().sum()


# test 데이터 채우기
## 숫자형 채우기
quantitative = house_test.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_test[col].fillna(house_train[col].mean(), inplace=True)
house_test[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_test.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_test[col].fillna("unknown", inplace=True)
house_test[qual_selected].isna().sum()


house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
# house_test.select_dtypes(include=[int, float])

df = pd.concat([house_train, house_test], ignore_index=True)
# df.info()
df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

## 이상치 탐색
train_df=train_df.query("GrLivArea <= 4500")

## train
train_x=train_df.drop("SalePrice", axis=1)
train_y=train_df["SalePrice"]

## test
test_x=test_df.drop("SalePrice", axis=1)

# Bootstrap 샘플 생성 함수
def bootstrap_data(train_x, train_y, size):
    idx = np.random.choice(np.arange(train_x.shape[0]), size, replace=True)
    return train_x.iloc[idx, :], train_y.iloc[idx]

# ElasticNet, RandomForest, 그리고 Stacking을 포함한 모델 훈련 과정
n_bootstraps = 10  # 부트스트랩 반복 횟수
train_x_stack_list = []
for i in range(n_bootstraps):
    print(f"Bootstrap iteration {i+1}")
    
    # 부트스트랩 샘플 생성
    bts_train_x, bts_train_y = bootstrap_data(train_x, train_y, size=1000)
    
    # ElasticNet 모델 그리드 서치
    eln_model = ElasticNet()
    param_grid_eln = {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'l1_ratio': [0, 0.1, 0.5, 1.0]
    }
    grid_search_eln = GridSearchCV(
        estimator=eln_model,
        param_grid=param_grid_eln,
        scoring='neg_mean_squared_error',
        cv=5
    )
    grid_search_eln.fit(bts_train_x, bts_train_y)
    best_eln_model = grid_search_eln.best_estimator_

    # RandomForest 모델 그리드 서치
    rf_model = RandomForestRegressor(n_estimators=100)
    param_grid_rf = {
        'max_depth': [3, 5, 7],
        'min_samples_split': [20, 10, 5],
        'min_samples_leaf': [5, 10, 20, 30],
        'max_features': ['sqrt', 'log2', None]
    }
    grid_search_rf = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid_rf,
        scoring='neg_mean_squared_error',
        cv=5
    )
    grid_search_rf.fit(bts_train_x, bts_train_y)
    best_rf_model = grid_search_rf.best_estimator_

    # 각 모델의 예측값을 새로운 학습 데이터로 사용
    y1_hat = best_eln_model.predict(train_x)
    y2_hat = best_rf_model.predict(train_x)

    # 스택킹을 위해 예측값을 저장
    train_x_stack_list.append(pd.DataFrame({
        f'y1_bootstrap_{i}': y1_hat,
        f'y2_bootstrap_{i}': y2_hat
    }))

# 각 부트스트랩 예측값을 합친 학습 데이터
train_x_stack = pd.concat(train_x_stack_list, axis=1)

# Ridge 모델을 블렌더로 사용하여 스택킹 학습
rg_model = Ridge()
param_grid_ridge = {'alpha': np.arange(0, 10, 0.01)}
grid_search_ridge = GridSearchCV(
    estimator=rg_model,
    param_grid=param_grid_ridge,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search_ridge.fit(train_x_stack, train_y)
blender_model = grid_search_ridge.best_estimator_

# 테스트 셋에 대해 예측 수행
test_x_stack_list = []
for i in range(n_bootstraps):
    pred_y_eln = best_eln_model.predict(test_x)
    pred_y_rf = best_rf_model.predict(test_x)
    test_x_stack_list.append(pd.DataFrame({
        f'y1_bootstrap_{i}': pred_y_eln,
        f'y2_bootstrap_{i}': pred_y_rf
    }))

test_x_stack = pd.concat(test_x_stack_list, axis=1)
final_predictions = blender_model.predict(test_x_stack)

# 결과 저장
sub_df["SalePrice"] = final_predictions
sub_df.to_csv("../../data/houseprice/sample_submission_bootstrap.csv", index=False)