import pandas as pd
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

data = pd.read_csv("realtor-data-cleaned3.csv")

scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)
df_standardized = pd.DataFrame(standardized_data, columns=data.columns)
X = df_standardized[['bed', 'acre_lot', 'zip_code', 'house_size', 'city']]
y = df_standardized['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)


X_train_sm = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_sm).fit()
print(model.summary())  # final regression model

# condidence intervals
confidence_intervals = model.conf_int(alpha=0.05)  # 95% confidence interval by default
confidence_intervals.columns = ['Lower Bound', 'Upper Bound']
confidence_intervals_df = pd.DataFrame(confidence_intervals,
                                       index=model.params.index,
                                       columns=['Lower Bound', 'Upper Bound'])
print("confidence_intervals for each attribute:", confidence_intervals_df)

table = PrettyTable()
table.title = "Feature Elimination"
table.field_names = ['Removed Feature', 'R^2', 'Adjusted R^2', 'AIC', 'BIC', 'MSE', 'p-value']
table.add_row(["NA", model.rsquared, model.rsquared_adj, model.aic, model.bic, model.mse_model, 'NA'])
table.float_format = '.3'

X_train_sm1 = X_train_sm.drop(['acre_lot'], axis=1)
model1 = sm.OLS(y_train, X_train_sm1).fit()
print(model1.summary())
table.add_row(
    ["acre_lot", model1.rsquared, model1.rsquared_adj, model1.aic, model1.bic, model1.mse_model, model.pvalues.max()])
print(table)

X_train_sm2 = X_train_sm1.drop(['city'], axis=1)
model2 = sm.OLS(y_train, X_train_sm2).fit()
print(model2.summary())
table.add_row(
    ["city", model2.rsquared, model2.rsquared_adj, model2.aic, model2.bic, model2.mse_model, model1.pvalues.max()])
print(table)

X_test_sm = sm.add_constant(X_test)
y_test_pred_std = model.predict(X_test_sm)
mean_y = np.mean(data['price'])
std_y = np.std(data['price'])
y_test_pred = y_test_pred_std * std_y + mean_y

y_train_pred_std = model.predict(X_train_sm)  # Predictions on training data
y_train_pred = y_train_pred_std * std_y + mean_y

print("prediction of dependent variable: \n", y_test_pred.head())

y_train_original = y_train * std_y + mean_y
y_test_original = y_test * std_y + mean_y

# Plotting
plt.figure(figsize=(12, 8))

plt.scatter(y_train_original/1e6, y_train_pred/1e6, color='blue', alpha=0.5, label='Train')
plt.scatter(y_test_original/1e6, y_test_pred/1e6, color='green', alpha=0.5, label='Test')

plt.xlabel('Actual Prices (Millions of Dollars)')
plt.ylabel('Predicted Prices (Millions of Dollars)')
plt.title('Actual vs Predicted Prices: Training and Testing Data')
plt.legend()

plt.show()
