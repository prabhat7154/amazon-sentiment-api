from pathlib import Path

import joblib

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


# Project folder ka path
BASE_DIR = Path(__file__).resolve().parent

# Model file ka path
MODEL_PATH = BASE_DIR / "model.pkl"


# FastAPI application
app = FastAPI(
    title="Amazon Review Sentiment API",
    description="Predicts whether an Amazon review is positive or negative."
)


# Trained model load karo
model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# Input ka structure
class PredictionRequest(BaseModel):

    reviewText: str

@app.get("/", response_class=HTMLResponse)
def home_page():

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title></title>
    </head>
    <body>
        <h1>Amazon Review Sentiment Predictor</h1>

        <p>Enter an Amazon review:</p>
        <textarea id="reviewText" placeholder="Enter your review here..."></textarea>
        <br>

        <button onclick="predictSentiment()">
            Predict Sentiment
        </button>

        <div id="result"></div>


        <script>

            async function predictSentiment() {
                const review = document.getElementById("reviewText").value;
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify({reviewText: review
                    })

                });

                const result = await response.json();
                document.getElementById("result").innerHTML =

                    "Prediction: " + result.prediction +
                    "<br>" +

                    "Sentiment: " + result.sentiment +
                    "<br>" +

                    "Positive Probability: " +
                    (result.positive_probability * 100) +
                    "%";

            }

        </script>

    </body>

    </html>
    """
@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "model_loaded": True
    }

@app.post("/predict")
def predict_sentiment(request: PredictionRequest):
    review = request.reviewText.strip()
    prediction = model.predict([review])
    prediction_value = int(prediction[0])
    probabilities = model.predict_proba([review])
    classes = model.classes_
    positive_position = list(classes).index(1)
    positive_probability = probabilities[0][positive_position]

    if prediction_value == 1:
        sentiment = "positive"
    else:
        sentiment = "negative"

    return {
        "prediction": prediction_value,
        "sentiment": sentiment,
        "positive_probability": round(float(positive_probability),2)

    }