import pandas as pd

df = pd.read_csv("news.csv")

# Separate FAKE and REAL
fake = df[df['label'] == 'FAKE']
real = df[df['label'] == 'REAL']

# Undersample FAKE
fake_under = fake.sample(len(real), random_state=42)

# Combine
df_balanced = pd.concat([fake_under, real]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df_balanced.to_csv("news_balanced.csv", index=False)

# Check counts
print(df_balanced['label'].value_counts())
