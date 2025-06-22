import joblib
import os
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from config.settings import MODEL_FILE
from services.data_service import DataService

class MLService:
    def __init__(self, data_service: DataService):
        self.model_pipeline: Optional[Pipeline] = None
        self.data_service = data_service
    
    def train_model(self, texts: List[str], labels: List[str]) -> Pipeline:
        """Train the sentiment classification model"""
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        pipeline.fit(texts, labels)
        return pipeline
    
    def load_or_create_model(self) -> None:
        """Load existing model or create new one with sample data"""
        if os.path.exists(MODEL_FILE):
            try:
                self.model_pipeline = joblib.load(MODEL_FILE)
                print("Model loaded from file")
                return
            except Exception as e:
                print(f"Error loading model: {e}, creating new one")
        
        print("Creating new model with training data")
        texts, labels = self.data_service.get_all_training_data()
        self.model_pipeline = self.train_model(texts, labels)
        self.save_model()
    
    def save_model(self) -> None:
        """Save the current model to file"""
        if self.model_pipeline:
            joblib.dump(self.model_pipeline, MODEL_FILE)
    
    def predict(self, text: str) -> str:
        """Predict sentiment for given text"""
        if not self.model_pipeline:
            raise ValueError("Model not loaded")
        
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        prediction = self.model_pipeline.predict([text])[0]
        return prediction
    
    def retrain_model(self) -> Dict[str, Any]:
        """Retrain model with all available data"""
        texts, labels = self.data_service.get_all_training_data()
        
        if len(texts) < 2:
            raise ValueError("Need at least 2 samples to train")
        
        self.model_pipeline = self.train_model(texts, labels)
        self.save_model()
        
        training_info = self.data_service.get_training_info()
        
        return {
            "message": "Model retrained successfully",
            **training_info
        }
    
    def train_new_model(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """Train model with new data (replaces existing training data)"""
        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have same length")
        
        if len(texts) < 2:
            raise ValueError("Need at least 2 samples to train")
        
        self.data_service.replace_training_data(texts, labels)
        self.model_pipeline = self.train_model(texts, labels)
        self.save_model()
        
        return {"message": f"Model trained successfully with {len(texts)} samples"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if not self.model_pipeline:
            raise ValueError("Model not loaded")
        
        classes = list(self.model_pipeline.classes_)
        training_info = self.data_service.get_training_info()
        
        return {
            "classes": classes,
            "feature_count": self.model_pipeline.named_steps['tfidf'].max_features,
            "model_type": "Logistic Regression with TF-IDF",
            **training_info
        }