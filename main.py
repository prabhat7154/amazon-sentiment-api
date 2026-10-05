from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.pkl"

app = FastAPI(
    title="Amazon Review Sentiment API",
    description="Predicts whether an Amazon review is positive or negative."
)

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


class PredictionRequest(BaseModel):

    reviewText: str


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