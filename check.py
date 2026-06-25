import pandas as pd

# Load the CSV
df = pd.read_csv("news.csv")

# Check the first few rows
print(df.head())

# Count the number of FAKE vs REAL
print(df['label'].value_counts())
