# PhaseI part 3: Random Forest Analysis

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df_encoded = pd.read_csv('realtor-data-encoded.csv')
print(f"The first 5 rows of the encoded dataset:\n{df_encoded.head()}")


print("Random Forest Analysis:")
X_rf = df_encoded.drop('price', axis=1)
y_rf = df_encoded['price']

X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=5805)

model = RandomForestRegressor(random_state=42)
model.fit(X_rf_train, y_rf_train)

importances = [round(imp, 3) for imp in model.feature_importances_]

feature_names = X_rf.columns

feature_importance_dict = dict(zip(feature_names, importances))
# print(feature_importance_dict)
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
print(sorted_feature_importance)