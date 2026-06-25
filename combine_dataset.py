import pandas as pd

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake['label'] = 'FAKE'
true['label'] = 'REAL'

# Combine
df = pd.concat([fake, true])

# Keep only needed columns
df = df[['text', 'label']]

# Save as news.csv
df.to_csv("news.csv", index=False)

print("✅ Combined dataset saved as news.csv")
