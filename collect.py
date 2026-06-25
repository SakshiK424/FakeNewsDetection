import pandas as pd

# Load True.csv with UTF-8 encoding (common for Kaggle datasets)
true_df = pd.read_csv("True.csv", encoding='utf-8')

# Show column names to check what’s inside
print(true_df.columns)
