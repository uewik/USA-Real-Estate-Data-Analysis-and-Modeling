# Phase I part 2: Binarization

import pandas as pd
from sklearn.preprocessing import LabelEncoder


df_cleaned = pd.read_csv('realtor-data-cleaned.csv')
print(f"The first 5 rows of the cleaned dataset:\n{df_cleaned.head()}")


print("Binarization:")
# One-hot encoding for 'status'
one_hot_encoded_status = pd.get_dummies(df_cleaned['status'])

# Label encoding for 'city' and 'state'
label_encoder_city = LabelEncoder()
label_encoded_city = label_encoder_city.fit_transform(df_cleaned['city'])

label_encoder_state = LabelEncoder()
label_encoded_state = label_encoder_state.fit_transform(df_cleaned['state'])

df_encoded = df_cleaned.copy()
df_encoded = df_encoded.drop(['status', 'city', 'state'], axis=1)
df_encoded = df_encoded.join(one_hot_encoded_status)
df_encoded['city'] = label_encoded_city
df_encoded['state'] = label_encoded_state

df_encoded = df_encoded.drop('ready_to_build', axis=1)
df_encoded['for_sale'] = df_encoded['for_sale'].map({True: 1, False: 0})

print(f"After encoding the dataset:\n{df_encoded.head()}")

df_encoded.to_csv('realtor-data-encoded.csv', index=False)  # save the encoded data to a new csv file
