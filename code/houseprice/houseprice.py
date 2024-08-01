import pandas as pd
import numpy as np

house_df=pd.read_csv("./data/houseprice/train.csv")
house_df.shape
house_df.head()
house_df.info()
price_mean=house_df["SalePrice"].mean()
price_mean

sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")
sub_df

sub_df["SalePrice"] = price_mean
sub_df

sub_df.to_csv("./data/houseprice/sample_submission.csv", index=False)


import numpy as np

np.random.seed(20240801)
np.random.choice(np.arange(7)+1, 7, replace=False)
np.random.choice(np.arange(4)+1, 1, replace=False)
