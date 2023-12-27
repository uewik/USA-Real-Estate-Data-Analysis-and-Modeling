# Covariance Matrix display through heatmap graph

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('realtor-data-cleaned2.csv')
df.drop(columns='city', inplace=True)

# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(df)

# df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

covariance_matrix = df.cov()

plt.figure(figsize=(10, 8))
sns.heatmap(covariance_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Covariance Matrix Heatmap After Scaling')
plt.show()