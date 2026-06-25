import requests

API_KEY = "20aadd4c72154e22864e2d9c519e8d83"   # Your API key
URL = "https://newsapi.org/v2/top-headlines"

params = {
    "country": "us",
    "apiKey": API_KEY
}

# ----------------------------
# Request API
# ----------------------------
response = requests.get(URL, params=params)

try:
    data = response.json()
except Exception as e:
    print("❌ JSON Decode Error:", e)
    exit()

# ----------------------------
# Handle API errors
# ----------------------------
if data.get("status") == "error":
    print("\n❌ API ERROR")
    print("Code    :", data.get("code"))
    print("Message :", data.get("message"))
    exit()

# ----------------------------
# Check if articles exist
# ----------------------------
if "articles" not in data:
    print("❌ ERROR: 'articles' not found in API response.")
    print("API Response:", data)
    exit()

articles = data["articles"]

# ----------------------------
# Print News
# ----------------------------
print("\n📰 TOP NEWS HEADLINES:\n")

if len(articles) == 0:
    print("No news found.")
else:
    for i, article in enumerate(articles, start=1):
        print(f"{i}. {article.get('title')}")
        print(f"   {article.get('url')}\n")
