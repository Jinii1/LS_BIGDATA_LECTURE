import pandas as pd

train = pd.read_csv('../../data/ml/chap7/train.csv', index_col='id') # 해당 열을 인덱스로
test = pd.read_csv('../../data/ml/chap7/test.csv', index_col='id')
submission = pd.read_csv('../../data/ml/chap7/sample_submission.csv', index_col='id')

train.shape, test.shape

train.info()
test.info()

train.head().T # 행과 열 위치 바꿔줘서 한눈에 보기 편하게

submission.head()

### 7.2.1 데이터 둘러보기

# 피처 요약본 만들기
def resumetable(df):
            print(f'데이터셋 형상: {df.shape}')
            summary = pd.DataFrame(df.dtypes, columns=['데이터 타입'])
            summary = summary.reset_index()
            summary = summary.rename(columns={'index': '피처'})
            summary['결측값 개수']= df.isnull().sum().values
            summary['고윳값 개수']= df.nunique().values
            summary['첫 번째 값']= df.loc[0].values
            summary['두 번째 값']= df.loc[1].values
            summary['세 번째 값']= df.loc[2].values

            return summary

resumetable(train)