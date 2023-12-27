import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("realtor-data-cleaned3.csv")

df1 = pd.DataFrame()
df1['bed'] = pd.cut(df['bed'], bins=[0, 3, 100], labels=['1-3', '5-99'], include_lowest=True)
df1['acre_lot'] = pd.cut(df['acre_lot'], bins=[0, 0.23, 100000], labels=['0-0.23', '0.235-100000'], include_lowest=True)
df1['house_size'] = pd.cut(df['house_size'], bins=[100, 1635, 1500000], labels=['0-1635', '1636-1500000'], include_lowest=True)

df_encoded = pd.DataFrame()
df_encoded['bed'] = (df1['bed'] == '1-3')
df_encoded['acre_lot'] = (df1['acre_lot'] == '0-0.23')
df_encoded['house_size'] = (df1['house_size'] == '0-1635')


frequent_itemsets = apriori(df_encoded, min_support=0.00001, use_colnames=True)


if frequent_itemsets.empty:
    print("No frequent itemsets found. Try adjusting the min_support value.")
else:
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

    print(rules.head(10).to_string())
