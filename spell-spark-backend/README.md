# SpellSpark Backend

## Overview

This project is the backend for SpellSpark, a machine learning-powered spelling growth recommender system for kids. It serves two models: a BiLSTM model for error classification and a CNN model for error prediction. The backend is built using FastAPI and provides an API for integration with a user interface.

## Project Structure

```
spell-spark-backend
├── models
│   ├── model1_bilstm.pth       # Serialized BiLSTM model for error classification
│   ├── model2_cnn.pth          # Serialized CNN model for error prediction
│   └── README.md               # Documentation for the models
├── src
│   ├── app.py                  # Entry point for the FastAPI application
│   ├── api
│   │   ├── endpoints.py        # API endpoints for model inference
│   │   └── __init__.py         # Initializes the API module
│   ├── ml
│   │   ├── train_models.py     # Code for training the machine learning models
│   │   ├── model1_bilstm.py    # BiLSTM model architecture and training logic
│   │   ├── model2_cnn.py       # CNN model architecture and training logic
│   │   └── __init__.py         # Initializes the ML module
│   └── utils
│       ├── logger.py           # Utility functions for logging
│       └── config.py           # Configuration settings for the application
├── data
│   ├── raw
│   │   └── vocabulary_bank.csv # Word bank for generating error profiles
│   └── final_pairs_with_error_labels_WIP_for_INSPECTIONs.xlsx # Training data
├── requirements.txt            # Project dependencies
└── .gitignore                  # Files and directories to ignore by Git
```

## Models

- **model1_bilstm.pth**: Serialized BiLSTM model trained for error classification. It predicts the error profile for a given misspelling.
- **model2_cnn.pth**: Serialized CNN model trained for error prediction. It generates error profiles for words in the word bank.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd spell-spark-backend
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models** (if not already trained):
   ```bash
   python src/ml/train_models.py
   ```

4. **Run the application**:
   ```bash
   uvicorn src.app:app --reload
   ```

5. **Access the API**:
   The API will be available at `http://127.0.0.1:8000` (or the configured port).

## API Endpoints

The backend provides the following endpoints:

1. **Root Endpoint**:
   - **URL**: `/`
   - **Method**: `GET`
   - **Description**: Returns a welcome message.

2. **Get Recommendations**:
   - **URL**: `/api/get-recommendations`
   - **Method**: `POST`
   - **Request Body**:
     ```json
     {
       "correct_word": "example",
       "user_misspelling": "exampel"
     }
     ```
   - **Response**:
     ```json
     {
       "error_profile": { ... },
       "recommended_words": ["sample", "example", "template"]
     }
     ```
   - **Description**: Predicts the error profile for the given misspelling and recommends similar words from the word bank.

## Usage

The backend is designed to:
- Train and save machine learning models (`model1` and `model2`).
- Serve the trained models via API endpoints.
- Generate error profiles for words in the word bank and use them for recommendations.

