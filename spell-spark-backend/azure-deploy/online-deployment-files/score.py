import os
import json
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import tokenizer  # Assuming tokenizer.py is in the same directory and contains CharTokenizer
import glob

# for collecting user inputs and model outputs
from azureml.ai.monitoring import Collector
from azureml.ai.monitoring.context import BasicCorrelationContext

# =====================================================================
# 1. MODEL ARCHITECTURE (Model 1)
# =====================================================================
class Model1_BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_layers=2, num_classes=5, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, num_classes)  

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        _, (h_n, _) = self.lstm(emb)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)  
        embeddings = F.relu(self.fc1(self.dropout(h)))  
        logits = self.fc2(embeddings)  
        return logits, embeddings

# =====================================================================
# 2. INITIALIZATION (Runs once when the endpoint starts)
# =====================================================================
def init():
    global model, config, vocab_bank_tensor, vocab_bank_words
    global inputs_collector, outputs_collector 
    
    # A. Load Model 1 using recursive glob search inside AZUREML_MODEL_DIR
    model_root = os.getenv('AZUREML_MODEL_DIR')
    
    # Find config and weights dynamically regardless of inner folder names
    config_path = glob.glob(os.path.join(model_root, '**', 'model1_config.json'), recursive=True)[0]
    weights_path = glob.glob(os.path.join(model_root, '**', 'model1_bilstm.pth'), recursive=True)[0]
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # Initialize and load weights
    model = Model1_BiLSTM(vocab_size=config['vocab_size'], num_classes=config['num_classes'])
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    
    # B. Load the Pre-computed Vocab Profiles CSV from the local script directory
    current_dir = os.path.dirname(__file__)
    csv_path = glob.glob(os.path.join(current_dir, '*vocab_profiles*.csv'))[0]
    vocab_df = pd.read_csv(csv_path)
        
    # Convert to a PyTorch tensor for fast similarity calculation
    vocab_bank_words = vocab_df['answer'].tolist()
    profiles_list = [json.loads(prof) for prof in vocab_df['predicted_profiles']]
    vocab_bank_tensor = torch.tensor(profiles_list, dtype=torch.float32)

    # INITIALIZE COLLECTORS ON ENDPOINT STARTUP
    inputs_collector = Collector(name='model_inputs')
    outputs_collector = Collector(name='model_outputs')

# =====================================================================
# 3. INFERENCE (Runs every time the API is called)
# =====================================================================
def run(raw_data):
    try:
        # Parse user input
        request = json.loads(raw_data)
        correct_word = request.get('correct_word', '')
        incorrect_word = request.get('incorrect_word', '')

        # Collect Input Data
        # Convert the input to a pandas DataFrame for logging
        input_df = pd.DataFrame([request])
        context = BasicCorrelationContext(id='spellspark_request')
        # Log the input
        context = inputs_collector.collect(input_df, context)

        
        # Tokenize using char2idx from config (or default alphabet mapping)
        char2idx = config.get('char2idx', None)
        pair_seq_ids = tokenizer.encode_pair(correct_word, incorrect_word, char2idx=char2idx)
        
        # Convert to PyTorch input tensor
        input_tensor = torch.tensor([pair_seq_ids], dtype=torch.long)
        
        # Run through Model 1
        with torch.no_grad():
            logits, _ = model(input_tensor)
            # Convert logits to soft probability profile
            user_profile = F.softmax(logits, dim=1)
            
        # Compute Cosine Similarity against the Vocab Bank
        similarities = cosine_similarity(user_profile, vocab_bank_tensor)
        
        # Extract the Top 3 Recommendations
        top_scores, top_indices = torch.topk(similarities, 3)
        
        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append({
                "word": vocab_bank_words[idx],
                "similarity": round(score, 4)
            })


        # Log outputs
        output_df = pd.DataFrame([{"recommendations": [res["word"] for res in results]}])
        outputs_collector.collect(output_df, context)
            
        # 4. Return successful JSON response
        return {"top_3_recommendations": results}
        
    except Exception as e:
        return {"error": str(e)}