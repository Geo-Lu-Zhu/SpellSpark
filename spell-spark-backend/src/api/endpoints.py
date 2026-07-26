from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np
import pandas as pd

router = APIRouter()

# Load models
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../../models')
model1_path = os.path.join(MODEL_DIR, 'model1_bilstm.pth')
model2_path = os.path.join(MODEL_DIR, 'model2_cnn.pth')

# In-memory storage for word bank profiles
word_bank_profiles = {}

class SpellingInput(BaseModel):
    correct_word: str
    user_misspelling: str

class PredictionResponse(BaseModel):
    error_profile: dict
    recommended_words: list

@router.on_event("startup")
async def load_word_bank_profiles():
    """Generate error profiles for all words in the word bank using model2."""
    global word_bank_profiles
    try:
        # Load the word bank from the CSV file
        word_bank_path = os.path.join(os.path.dirname(__file__), '../../data/raw/vocabulary_bank.csv')
        word_bank = pd.read_csv(word_bank_path, header=None, names=["word"])
        
        # Generate error profiles using model2
        model2 = joblib.load(model2_path)
        for word in word_bank["word"]:
            error_profile = model2.predict([word])[0]
            word_bank_profiles[word] = error_profile
        print(f"Loaded error profiles for {len(word_bank_profiles)} words in the word bank.")
    except Exception as e:
        print(f"Error loading word bank profiles: {e}")

@router.post("/get-recommendations", response_model=PredictionResponse)
async def get_recommendations(input_data: SpellingInput):
    try:
        # Load model1
        model1 = joblib.load(model1_path)
        
        # Predict error profile using model1
        error_profile = model1.predict([input_data.correct_word, input_data.user_misspelling])[0]
        
        # Find recommendations based on cosine similarity
        recommendations = []
        for word, profile in word_bank_profiles.items():
            similarity = np.dot(error_profile, profile) / (np.linalg.norm(error_profile) * np.linalg.norm(profile))
            recommendations.append((word, similarity))
        
        # Sort recommendations by similarity
        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:5]
        recommended_words = [word for word, _ in recommendations]
        
        return PredictionResponse(error_profile=error_profile, recommended_words=recommended_words)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))