from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model & vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]
    news_vector = vectorizer.transform([news])
    output = model.predict(news_vector)[0]
    return render_template("index.html", result=output)

if __name__ == "__main__":
    app.run(debug=True)
