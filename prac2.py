import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project1/proj1/bs.csv")
df

before_name = []
after_name = []
lst_mean12_19 = []

for i in range(12, 23):
  before_name.append("20" + str(i))
  before_name.append("20" + str(i) + ".1")
  before_name.append("20" + str(i) + ".2")
  before_name.append("20" + str(i) + ".3")
  before_name.append("20" + str(i) + ".4")
  before_name.append("20" + str(i) + ".5")
  before_name.append("20" + str(i) + ".6")
  after_name.append(str(i) + "_1519")
  after_name.append(str(i) + "_2024")
  after_name.append(str(i) + "_2529")
  after_name.append(str(i) + "_3034")
  after_name.append(str(i) + "_3539")
  after_name.append(str(i) + "_4044")
  after_name.append(str(i) + "_4549")

br = df.copy()
br.drop(0, inplace = True)
br.reset_index(drop=True, inplace=True)

for i in range(0, len(before_name)):
  br.rename(columns = {before_name[i] : after_name[i]}, inplace = True)

br[after_name] = br[after_name].apply(pd.to_numeric)

br = br.assign(
  mean20 = (br["20_2024"] + br["20_2529"] + br["20_3034"]) / 3,
  mean21 = (br["21_2024"] + br["21_2529"] + br["21_3034"]) / 3,
  mean22 = (br["22_2024"] + br["22_2529"] + br["22_3034"]) / 3)

# 2012 ~ 2019년까지의 평균은 df에 열로 추가 안 하고, List 형태로 보관 (각 요소는 Series)
for i in range(0, 56, 7):
  lst_mean12_19.append((br[after_name[i + 1]] + br[after_name[i + 2]] + br[after_name[i + 3]]) / 3)

# 2012 ~ 2019년의 large, small 계산 (List에 있는 Series끼리 비교를 해서 결과를 df의 열로 추가한다)
for i in range(12, 20):
  br["compare" + str(i)] = np.where(lst_mean12_19[i - 12].mean() <= lst_mean12_19[i - 12], "large", "small")

# 2020 ~ 2022년의 평균은 df에 있으므로 계산 후 열 추가
br["compare20"] = np.where(br["mean20"].mean() <= br["mean20"], "large", "small")
br["compare21"] = np.where(br["mean21"].mean() <= br["mean21"], "large", "small")
br["compare22"] = np.where(br["mean22"].mean() <= br["mean22"], "large", "small")

pd.set_option('display.max.rows', None)
pd.set_option('display.max.columns', None)

br3 = br.iloc[[0]]
# type(br3["21_2024"][0]) #데이터프레임으로 나오는지 확인하기 위한 코드
br3 = br3.iloc[:, 57:] # 20-22년도 데이터 추출

br3 = br3.transpose()
br3

br3.info()

br3 = br3.rename(columns = {0 : 'birth_rate'})
br3 = br3.rename_axis(columns = {'' : 'year'}, index = None)
br3
br3 = br3.reset_index().rename(columns={'index': 'year'})

br3['number'] = np.where(br3['year']\
                  .isin(['20_2024', '20_2529', '20_3034', '21_2024', '21_2529', '21_3034', '22_2024', '22_2529', '22_3034']), '1', '2')


br3['number'] = br3['number'].apply(pd.to_numeric)

br3_youth_rate = br3.query('number == 1')['birth_rate'].mean()
br3_non_youth_rate = br3.query('number == 2')['birth_rate'].mean()

br3_20 = br3[:7] # 20년도 데이터프레임
br3_20

br3.plot.bar(rot=0)

!pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(data = br3_20, x = 'year', y = 'rate')
plt.show()
plt.clf()



# 연진






# 연예진
br3.groupby(['year'])

br3 = br3.transpose()
br3 = br3.rename(row={'index': 'year'})
br3.query(')



br3['new_number'] = np.where(br3['year']\
                  .isin(['20_1519', '20_2529', '20_3034', '20_2024', '20_3539', '20_4044 ', '20_4549']), '2020')

br3.groupby('year')[['birthrate[1:4]', 'birthrate[0] + birate[4:7]' ]].mean()
