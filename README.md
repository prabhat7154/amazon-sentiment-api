# What model predicts
The model predicts whether an Amazon review is positive(1) or negative(0).


# Request Body for prediction(POST:/predict):
{
  "reviewText": "This is a fantastic app. My kids love it."
}

# How to run locally:
In Terminal:
1)pip install -r requirements.txt
2)uvicorn main:app --reload