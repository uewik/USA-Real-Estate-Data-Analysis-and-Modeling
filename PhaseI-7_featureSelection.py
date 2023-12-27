# Feature Selection based on Random Forest Analysis

import pandas as pd

df_encoded = pd.read_csv('realtor-data-encoded.csv')
df_fs = df_encoded.drop(columns=['state', 'for_sale'])

df_fs.to_csv('realtor-data-feature-selected.csv', index=False)
