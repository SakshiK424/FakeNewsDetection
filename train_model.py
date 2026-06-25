import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# 1️⃣ Load the dataset
df = pd.read_csv("news_balanced.csv")

# 2️⃣ Split features and labels
X = df['text']      # news content
y = df['label']     # FAKE or REAL

# 3️⃣ Convert text to numeric vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# 4️⃣ Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_vec, y)

# 5️⃣ Save trained model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Naive Bayes model trained and saved as model.pkl & vectorizer.pkl")
