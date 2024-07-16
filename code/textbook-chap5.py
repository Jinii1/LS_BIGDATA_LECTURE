import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 탐색 함수
# head()
# tail()
# shape # method vs. attribute
# info()
# describe()

exam = pd.read_csv('data/exam.csv')
exam.head()
exam.tail(10)
exam.shape
exam.info()
exam.describe()

exam2 = exam.copy()
exam2 = exam2.rename(columns = {'nclass' :'class'})
exam2

exam2['total'] = exam2['math'] + exam2['english'] + exam2['science']

exam2['test'] = np.where(exam2['total'] >= 200, 'pass', 'fail')
exam2

exam2['test'].value_counts().plot.bar(rot =0)
plt.show()
plt.clf()

exam2['test2'] = np.where(exam2['total'] >= 200, 'A',
                 np.where(exam2['total'] >= 100, 'B', 'C'))
exam2['test2']













