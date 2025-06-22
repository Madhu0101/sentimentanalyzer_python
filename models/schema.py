from pydantic import BaseModel
from typing import List

class FeedbackRequest(BaseModel):
    text: str

class FeedbackResponse(BaseModel):
    text: str
    sentiment: str

class TrainingData(BaseModel):
    texts: List[str]
    labels: List[str]

class PredictionFeedback(BaseModel):
    text: str
    predicted_sentiment: str
    actual_sentiment: str