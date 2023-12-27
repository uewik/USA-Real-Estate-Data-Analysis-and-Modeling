# VIF

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

df_encoded = pd.read_csv('realtor-data-encoded.csv')


def calculate_vif(df):
    df = add_constant(df)
    vifs = pd.Series([variance_inflation_factor(df.to_numpy(), i) for i in range(df.shape[1])], index=df.columns)
    return vifs


df_feature = df_encoded.drop(columns='price')

vif_threshold = 5

high_vif = True

vif_data = pd.Series()
while high_vif:
    vif_data = calculate_vif(df_feature)
    vif_data.drop(labels='const', inplace=True)
    if vif_data.max() > vif_threshold:
        feature_to_remove = vif_data.idxmax()
        df_feature.drop(columns=feature_to_remove, inplace=True)
    else:
        high_vif = False

print(f'VIF for each feature:\n{vif_data}')
