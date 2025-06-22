import json
import os
from datetime import datetime
from typing import List, Tuple, Dict, Any
from config.settings import TRAINING_HISTORY_FILE, SAMPLE_DATA

class DataService:
    def __init__(self):
        self.training_history = []
        self.load_training_history()
    
    def save_training_history(self) -> None:
        """Save training history to file"""
        try:
            with open(TRAINING_HISTORY_FILE, 'w') as f:
                json.dump(self.training_history, f, indent=2)
        except Exception as e:
            print(f"Error saving training history: {e}")
    
    def load_training_history(self) -> None:
        """Load training history from file"""
        try:
            if os.path.exists(TRAINING_HISTORY_FILE):
                with open(TRAINING_HISTORY_FILE, 'r') as f:
                    self.training_history = json.load(f)
                print(f"Loaded {len(self.training_history)} training records")
            else:
                # Initialize with sample data
                self.training_history = [
                    {
                        "text": text,
                        "label": label,
                        "source": "sample_data",
                        "timestamp": datetime.now().isoformat()
                    }
                    for text, label in zip(SAMPLE_DATA["texts"], SAMPLE_DATA["labels"])
                ]
                self.save_training_history()
        except Exception as e:
            print(f"Error loading training history: {e}")
            self.training_history = []
    
    def get_all_training_data(self) -> Tuple[List[str], List[str]]:
        """Get all training data (sample + historical predictions)"""
        texts = []
        labels = []
        
        for record in self.training_history:
            texts.append(record["text"])
            labels.append(record["label"])
        
        return texts, labels
    
    def add_prediction_feedback(self, text: str, predicted_sentiment: str, actual_sentiment: str) -> Dict[str, Any]:
        """Add prediction feedback to training history"""
        new_record = {
            "text": text,
            "label": actual_sentiment,
            "source": "user_feedback",
            "timestamp": datetime.now().isoformat(),
            "predicted_sentiment": predicted_sentiment
        }
        
        self.training_history.append(new_record)
        self.save_training_history()
        
        return {
            "message": "Prediction added to training data successfully",
            "total_training_samples": len(self.training_history)
        }
    
    def replace_training_data(self, texts: List[str], labels: List[str]) -> None:
        """Replace all training history with new data"""
        self.training_history = [
            {
                "text": text,
                "label": label,
                "source": "manual_training",
                "timestamp": datetime.now().isoformat()
            }
            for text, label in zip(texts, labels)
        ]
        self.save_training_history()
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get information about training data sources"""
        sample_count = sum(1 for record in self.training_history if record.get("source") == "sample_data")
        user_feedback_count = sum(1 for record in self.training_history if record.get("source") == "user_feedback")
        manual_training_count = sum(1 for record in self.training_history if record.get("source") == "manual_training")
        
        return {
            "total_training_samples": len(self.training_history),
            "data_sources": {
                "sample_data": sample_count,
                "user_feedback": user_feedback_count,
                "manual_training": manual_training_count
            }
        }
    
    def get_training_history(self, limit: int = 50) -> Dict[str, Any]:
        """Get training history with optional limit"""
        return {
            "total_records": len(self.training_history),
            "history": self.training_history[-limit:] if limit else self.training_history
        }
    
    def reset_to_sample_data(self) -> Dict[str, Any]:
        """Reset training history to sample data only"""
        self.training_history = [
            {
                "text": text,
                "label": label,
                "source": "sample_data",
                "timestamp": datetime.now().isoformat()
            }
            for text, label in zip(SAMPLE_DATA["texts"], SAMPLE_DATA["labels"])
        ]
        self.save_training_history()
        
        return {
            "message": "Training history reset to sample data",
            "total_samples": len(self.training_history)
        }