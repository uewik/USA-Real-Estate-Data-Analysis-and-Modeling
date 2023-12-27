# Sample Pearson Correlation coefficients Matrix display through heatmap graph

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('realtor-data-cleaned2.csv')
df_numeric = df.drop(columns='city')

df_cm = df_numeric.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Pearson Correlation Coefficients Matrix')
plt.show()

df.drop(columns='bath', inplace=True)

df.to_csv('realtor-data-cleaned3.csv', index=False)
