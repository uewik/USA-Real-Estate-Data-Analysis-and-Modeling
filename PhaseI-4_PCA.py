# Principal Component Analysis and condition number:

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np

df_encoded = pd.read_csv('realtor-data-encoded.csv')

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_encoded)

pca = PCA()
pca.fit(scaled_data)

total_variance = np.cumsum(pca.explained_variance_ratio_)
# n_components = np.where(total_variance >= 0.90)[0][0] + 1
n_components = np.argmax(total_variance >= 0.9) + 1

print(f"Number of components explaining >= 90% variance: {n_components}")

# Apply PCA with selected number of components
pca = PCA(n_components=n_components)
pca.fit(scaled_data)  # Changed here

condition_number = np.linalg.cond(pca.components_)
print('Condition Number:', condition_number.round(3))

# Map components back to original features
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_components)]
most_important_names = [df_encoded.columns[most_important[i]] for i in range(n_components)]
print("Most important features:", most_important_names)