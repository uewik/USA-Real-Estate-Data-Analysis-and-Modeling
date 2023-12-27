# phase I part 1: Data cleaning and Check for data duplications and removal

import pandas as pd

df_original = pd.read_csv('realtor-data.csv')

print(f"The first 5 rows of the original dataset:\n{df_original.head().to_string()}")

print("The number of records in the original dataset: ", df_original.shape[0])

print(f"The number of missing data in each feature:\n{df_original.isnull().sum()}")


print("Data cleaning:")
df_cleaned = df_original.dropna(subset='price')  # remove any rows with missing 'price' data
print("After removing rows with missing price, the number of records left: ", df_cleaned.shape[0])

df_cleaned = df_cleaned.drop('prev_sold_date', axis=1)  # remove the column 'prev_sold_date'

# Drop records with 'house_size', 'bed', 'bath' and 'acre_lot' are all missing
df_cleaned = df_cleaned.dropna(subset=['bed', 'bath', 'acre_lot', 'house_size'], how='all')
print("After removing rows with missing house info, the number of records left: ", len(df_cleaned.index))

df_cleaned.dropna(subset='zip_code', inplace=True)  # Drop records with missing 'zip_code'
print("After removing rows with missing zip code, the number of records left: ", len(df_cleaned.index))

# fill the missing values in 'bed', 'bath', 'acre_lot' with the median of each column by the same 'zip_code'
df_cleaned['bed'] = df_cleaned.groupby('zip_code')['bed'].transform(lambda x: x.fillna(x.median()))
df_cleaned.dropna(subset='bed', inplace=True)  # Drop records with missing 'bed' in certain 'zip_code'
df_cleaned['bath'] = df_cleaned.groupby('zip_code')['bath'].transform(lambda x: x.fillna(x.median()))
df_cleaned.dropna(subset='bath', inplace=True)
df_cleaned['acre_lot'] = df_cleaned.groupby('zip_code')['acre_lot'].transform(lambda x: x.fillna(x.median()))
df_cleaned.dropna(subset='acre_lot', inplace=True)
print("After cleaning missing data in bed, bath and acre_lot column, the number of records left: "
      , len(df_cleaned.index))

df_cleaned.dropna(subset='city', inplace=True)  # Drop records with missing 'city' data

# fill the missing values in 'house_size' with the median of each column first by the same 'zip_code' and then by the
# same 'bed' and 'bath'
df_cleaned['house_size'] = (df_cleaned.groupby(['zip_code', 'bed', 'bath'])['house_size']
                            .transform(lambda x: x.fillna(x.median())))
df_cleaned.dropna(subset='house_size', inplace=True)

print(f"check if there is any missing data left:\n{df_cleaned.isnull().sum()}")
print("After cleaning all missing data, the number of records left: ", len(df_cleaned.index))


print("Check for data duplications and removal:")
print("The number of duplicated records: ", df_cleaned.duplicated().sum())

df_cleaned.drop_duplicates(inplace=True)  # remove duplicated records
print("After removing duplicated records, the number of records left: ", len(df_cleaned.index))

df_cleaned.to_csv('realtor-data-cleaned.csv', index=False)  # save the cleaned data to a new csv file



