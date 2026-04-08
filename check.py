import pandas as pd

df = pd.read_csv("data/telco_churn_100k.csv")

print("Columns are:")
print(df.columns)

print("\nFirst rows:")
print(df.head())