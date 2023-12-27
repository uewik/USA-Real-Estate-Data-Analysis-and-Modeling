# Singular Value Decomposition Analysis:

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df_encoded = pd.read_csv('realtor-data-encoded.csv')

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_encoded)

U, S, VT = np.linalg.svd(scaled_data, full_matrices=False)

feature_contributions = np.abs(VT).sum(axis=0)
feature_names = df_encoded.columns
feature_importance = pd.Series(feature_contributions, index=feature_names).sort_values(ascending=False)

print(f"Feature importance:\n{feature_importance}")