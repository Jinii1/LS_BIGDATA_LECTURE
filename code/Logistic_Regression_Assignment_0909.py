# 종속변수: 백혈병 세포 관측 불가 여부 (REMISS), 1이면 관측 안됨을 의미

# 독립변수:
# 골수의 세포성 (CELL)
# 골수편의 백혈구 비율 (SMEAR)
# 골수의 백혈병 세포 침투 비율 (INFIL)
# 골수 백혈병 세포의 라벨링 인덱스 (LI)
# 말초혈액의 백혈병 세포 수 (BLAST)
# 치료 시작 전 최고 체온 (TEMP)

## 문제 1.
## 데이터를 로드하고, 로지스틱 회귀모델을 적합하고, 회귀 표를 작성하세요.

# 데이터 로드
import pandas as pd
file_path = '../data/leukemia_remission.txt'
df = pd.read_csv(file_path, sep='\t')

# 로지스틱 회귀모델 적합, 회귀 표 작성
import statsmodels.api as sm
model = sm.formula.logit("REMISS ~ CELL + SMEAR + INFIL + LI + BLAST + TEMP", data=df).fit()
print(model.summary())

## 문제 2.
## 해당 모델은 통계적으로 유의한가요? 그 이유를 검정통계량를 사용해서 설명하시오.
개별 변수들은 대부분 유의하지 않으며, 특히 p-value가 0.05보다 큰 값들을 보입니다.
LI 변수는 p-value가 0.101로 10% 유의수준에서는 유의할 수 있으나, 엄격한 5% 유의수준에서는 유의하지 않습니다.

# 문제 3.
# 유의수준이 0.2를 기준으로 통계적으로 유의한 변수는 몇개이며, 어느 변수 인가요?
2개 (LI, TEMP)


# 문제 4. 다음 환자에 대한 오즈는 얼마인가요?
# CELL (골수의 세포성): 65%
# SMEAR (골수편의 백혈구 비율): 45%
# INFIL (골수의 백혈병 세포 침투 비율): 55%
# LI (골수 백혈병 세포의 라벨링 인덱스): 1.2
# BLAST (말초혈액의 백혈병 세포 수): 1.1세포/μL
# TEMP (치료 시작 전 최고 체온): 0.9

import numpy as np
log_odds=64.2581 + 30.8301*0.65 + 24.6863*0.45 + -24.9745*0.55 + 4.3605*1.2 + -0.0115*1.1 + -100.1734*0.9
odds = np.exp(log_odds) # 0.03817459641135519


## 문제 5. 위 환자의 혈액에서 백혈병 세포가 관측되지 않은 확률은 얼마인가요?
p_hat = odds / (odds + 1)
1 - p_hat # 0.9632291171992526


# 문제 6. TEMP 변수의 계수는 얼마이며, 해당 계수를 사용해서 TEMP 변수가 백혈병 치료에 대한 영향을 설명하시오.
-100.1734
체온이 증가할수록 백혈병 치료에 부정적인 영향을 미칩니다
체온이 1도 상승할 때마다 백혈병 세포가 관측되지 않을 가능성은 급격히 줄어들게 됩니다.
이는 고열이 백혈병 치료 성공에 방해가 될 수 있음을 의미합니다.

## 문제 7. CELL 변수의 99% 오즈비에 대한 신뢰구간을 구하시오.
# 99% 오즈비 신뢰구간 계산
odds_ratios = np.exp(model.params)
conf = model.conf_int()
conf = np.exp(conf)

# CELL 변수의 신뢰 구간
cell_confidence_interval = conf.loc['CELL']
print(f"CELL 변수의 99% 오즈비 신뢰구간: {cell_confidence_interval}")

#CELL 변수의 99% 오즈비 신뢰구간: 
#0    1.027206e-31
#1    5.847804e+57




문제 8. 주어진 데이터에 대하여 로지스틱 회귀 모델의 예측 확률을 구한 후, 50% 이상인 경우 1로 처리하여, 혼동 행렬를 구하시오.
pred_probs = model.predict(X)
predictions = (pred_probs >= 0.5).astype(int)

# 혼동 행렬
conf_matrix = confusion_matrix(y, predictions)



문제 9. 해당 모델의 Accuracy는 얼마인가요?
accuracy = accuracy_score(y, predictions)


문제 10. 해당 모델의 F1 Score를 구하세요.
f1 = f1_score(y, predictions)