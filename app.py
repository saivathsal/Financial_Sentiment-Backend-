from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
from analyze import analyze_sentiment

app = FastAPI(title="Sentiment Analysis API")

class TextInput(BaseModel):
    texts: List[str]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API"}

@app.post("/analyze")
def analyze(text_input: TextInput):
    df = analyze_sentiment(text_input.texts)
    return df.to_dict(orient="records")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000)