from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "amazon_dataset.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

df = pd.read_csv(DATA_PATH)


X = df["reviewText"].astype(str)
y = df["Positive"].astype(int)

tfidf = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=100000
)

classifier = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model = Pipeline([
    ("tfidf", tfidf),
    ("classifier", classifier)
])

model.fit(X, y)

joblib.dump(model, MODEL_PATH)

