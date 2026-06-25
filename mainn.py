from flask import Flask, render_template, request
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F

app = Flask(__name__)

# Load tokenizer and model (use local folder 'distilbert_model')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert_model')
model = DistilBertForSequenceClassification.from_pretrained('distilbert_model')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form["news"]

    inputs = tokenizer(news_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()

    result = "REAL" if pred == 1 else "FAKE"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
