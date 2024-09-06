from sklearn.metrics import confusion_matrix
import numpy as np

# 아델리: 'A'
# 친스트랩(아델리 아닌것): 'C'

y_true = np.array(['A', 'A', 'C', 'A', 'C', 'C', 'C'])
y_pred = np.array(['A', 'C', 'A', 'A', 'A', 'C', 'C'])

conf_mat=confusion_matrix(y_true=y_true, 
                 y_pred=y_pred,
                 labels=["A", "C"])
conf_mat

from sklearn.metrics import ConfusionMatrixDisplay

p=ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                        display_labels=('Adelie', 'Chinstrap'))
p.plot(cmap='Blues')

# 2, 1, 2, 2가 어떤 의미일까?
# True label: 실제값(Adeli, Chinstrap)
# Predicted label: 모델에서 데이터 포인트가 특정 종이라고 예측한 그 label
좌측 상단 2: 실제 값이 adelie인 경우 adelie라 맞춘게 2개인것
좌측 하단 2: 실제 값이 chinstrap인데 adelie로 예측한 경우
우측 상단 1: 실제 값이 adelie 인데 chinstrap으로 예측한 경우
우측 하단 2: 

# y_true=(A, A, C, A, C, C, C)
# y_pred2 = (C, A, A, A, C, C, C)
# 혼동행렬 만들어보세요!
