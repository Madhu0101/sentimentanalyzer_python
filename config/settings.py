import os
from datetime import datetime

# File paths
MODEL_FILE = "sentiment_model.pkl"
TRAINING_HISTORY_FILE = "training_history.json"

# CORS settings
CORS_ORIGINS = ["http://localhost:3000"]

# Sample training data for initial model
SAMPLE_DATA = {
    "texts": [
        "I love this product! It's amazing and works perfectly.",
        "This is the worst service I've ever experienced.",
        "The product is okay, nothing special but does the job.",
        "Absolutely fantastic! Exceeded my expectations.",
        "Terrible quality, waste of money.",
        "Good value for money, satisfied with purchase.",
        "Outstanding customer service, very helpful staff.",
        "Product broke after one day, very disappointed.",
        "Average product, could be better but acceptable.",
        "Excellent quality and fast delivery!",
        "Poor communication from support team.",
        "Really happy with this purchase, recommend it!",
        "Not worth the price, found better alternatives.",
        "Great experience overall, will buy again.",
        "Product didn't match description, returning it.",
        "It works as expected, no complaints.",
        "The delivery was on time, product is standard.",
        "Normal experience, nothing to highlight.",
        "Product arrived in good condition, as described.",
        "Standard quality, meets basic requirements.",
        "The item is functional, does what it's supposed to do.",
        "Received the order, everything seems fine.",
        "Product is adequate for the price point.",
        "No issues with the purchase, works normally.",
        "Basic functionality, nothing special or bad."
    ],
    "labels": [
        "positive", "negative", "neutral", "positive", "negative",
        "positive", "positive", "negative", "neutral", "positive",
        "negative", "positive", "negative", "positive", "negative",
        "neutral", "neutral", "neutral", "neutral", "neutral",
        "neutral", "neutral", "neutral", "neutral", "neutral"
    ]
}