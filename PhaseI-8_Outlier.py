# Anomaly detection/Outlier Analysis and removal

import pandas as pd
from scipy import stats
import numpy as np

df_fs = pd.read_csv('realtor-data-feature-selected.csv')

df_fs['price_z_score'] = np.abs(stats.zscore(df_fs['price']))

z_score_threshold = 3

df_fs_cleaned = df_fs[df_fs['price_z_score'] <= z_score_threshold]

df_fs_cleaned = df_fs_cleaned.drop(columns=['price_z_score'])

print("the number of outlier removed:", df_fs.shape[0] - df_fs_cleaned.shape[0])

print("After removing outliers, the number of records in the data is", df_fs_cleaned.shape[0])

df_fs_cleaned.to_csv('realtor-data-cleaned2.csv', index=False)
