import datetime
import pandas as pd
from model import predict

def analyze_sentiment(texts):
    now = datetime.datetime.now()
    results = []
    for text in texts:
        sentiment, conf = predict(text)
        score = int(round(conf * 10))
        results.append({
            "Datetime": now,
            "Text": text,
            "Sentiment": sentiment.capitalize(),
            "Confidence": round(conf, 2),
            "Sentiment_Score": score
        })
    return pd.DataFrame(results)
