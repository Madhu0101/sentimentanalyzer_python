from fastapi import APIRouter, HTTPException
from models.schema import FeedbackRequest, FeedbackResponse, TrainingData, PredictionFeedback
from services.ml_service import MLService
from services.data_service import DataService

# Initialize services
data_service = DataService()
ml_service = MLService(data_service)

router = APIRouter()

@router.get("/")
async def root():
    return {"message": "Sentiment Analysis API is running!"}

@router.post("/predict", response_model=FeedbackResponse)
async def predict_sentiment(request: FeedbackRequest):
    """Predict sentiment for given text"""
    try:
        prediction = ml_service.predict(request.text)
        return FeedbackResponse(
            text=request.text,
            sentiment=prediction
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/add-prediction-to-training")
async def add_prediction_to_training(feedback: PredictionFeedback):
    """Add a prediction result to training data for future model improvement"""
    try:
        result = data_service.add_prediction_feedback(
            feedback.text,
            feedback.predicted_sentiment,
            feedback.actual_sentiment
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding to training data: {str(e)}")

@router.post("/retrain-model")
async def retrain_model():
    """Retrain model with all available data (sample + user feedback)"""
    try:
        result = ml_service.retrain_model()
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@router.post("/train")
async def train_new_model(training_data: TrainingData):
    """Train model with new data (this will replace all existing training data)"""
    try:
        result = ml_service.train_new_model(training_data.texts, training_data.labels)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@router.get("/model/info")
async def get_model_info():
    """Get information about the current model"""
    try:
        return ml_service.get_model_info()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training-history")
async def get_training_history():
    """Get all training history"""
    return data_service.get_training_history()

@router.delete("/training-history")
async def reset_training_history():
    """Reset training history to sample data only"""
    try:
        result = data_service.reset_to_sample_data()
        # Retrain model with sample data only
        ml_service.retrain_model()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset error: {str(e)}")

# Function to initialize ML service (called from main.py)
def initialize_ml_service():
    """Initialize the ML service - called during startup"""
    ml_service.load_or_create_model()