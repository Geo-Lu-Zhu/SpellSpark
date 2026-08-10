import azure.functions as func
import logging
import os
import json
import tempfile
import glob
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
import tokenizer  

app = func.FunctionApp()

# =====================================================================
# 1. MODEL ARCHITECTURE
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
# 2. COLD START INITIALIZATION (Runs once per instance lifecycle)
# =====================================================================
logging.info("Initializing model dynamically from Azure ML...")

try:
    credential = DefaultAzureCredential()
    
    # Fetch the subscription ID from environment variables
    sub_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    if not sub_id:
        raise ValueError("AZURE_SUBSCRIPTION_ID environment variable is missing!")
    
    # Initialize the MLClient targeting your TestML workspace
    ml_client = MLClient(
        credential,
        subscription_id=sub_id,
        resource_group_name="test1",                            
        workspace_name="TestML"                                 
    )
    
    temp_dir = tempfile.gettempdir()
    logging.info("Downloading model artifacts from Azure ML...")
    
    # Download the registered model artifacts to the temp directory
    ml_client.models.download(
        name="model1_bilstm_teacher",
        version="3",
        download_path=temp_dir
    )
    
    config_path = glob.glob(os.path.join(temp_dir, '**', 'model1_config.json'), recursive=True)[0]
    weights_path = glob.glob(os.path.join(temp_dir, '**', 'model1_bilstm.pth'), recursive=True)[0]
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    model = Model1_BiLSTM(vocab_size=config['vocab_size'], num_classes=config['num_classes'])
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    
    logging.info("Model loaded successfully from Azure ML.")
    
except Exception as e:
    logging.error(f"Failed to download or load model: {str(e)}")

# B. Load Pre-computed Profiles
datastore_path = os.getenv("DATASTORE_URI")
if not datastore_path:
    logging.warning("DATASTORE_URI not set. Local fallback not implemented.")
else:
    try:
        vocab_df = pd.read_csv(datastore_path, storage_options={"credential": credential})
        vocab_bank_words = vocab_df['answer'].tolist()
        profiles_list = [json.loads(prof) for prof in vocab_df['predicted_profiles']]
        vocab_bank_tensor = torch.tensor(profiles_list, dtype=torch.float32)
        logging.info("Successfully loaded vocabulary profiles.")
    except Exception as e:
        logging.error(f"Failed to load vocab from datastore: {str(e)}")

# =====================================================================
# 3. HTTP TRIGGER (Runs on every API request)
# =====================================================================
@app.route(route="score", auth_level=func.AuthLevel.FUNCTION)
def score_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Input schema:
        # {"requests": [{"correct_word": "...", "incorrect_word": "..."}, ...], "debug": false}
        req_body = req.get_json()
        items = req_body.get('requests', [])
        debug = bool(req_body.get('debug', False))

        if not items:
            return func.HttpResponse(
                json.dumps({"error": "Payload must contain 'requests': a non-empty list of {correct_word, incorrect_word} objects."}),
                mimetype="application/json",
                status_code=400
            )

        char2idx = config.get('char2idx', None)
        idx_to_class = config.get('idx_to_class', {})

        # Run each input individually — no padding, matches single-input behavior
        all_logits = []
        all_embeddings = []
        with torch.no_grad():
            for item in items:
                seq_ids = tokenizer.encode_pair(
                    str(item.get('correct_word', '')),
                    str(item.get('incorrect_word', '')),
                    char2idx=char2idx
                )
                input_tensor = torch.tensor([seq_ids], dtype=torch.long)
                logits, embedding = model(input_tensor)  # [1, num_classes], [1, 64]
                all_logits.append(logits)
                all_embeddings.append(embedding)

        user_embeddings = torch.cat(all_embeddings, dim=0)  # [B, 64]
        logits = torch.cat(all_logits, dim=0)               # [B, num_classes]

        # Matrix cosine similarity: [B, N]
        user_norm = F.normalize(user_embeddings, p=2, dim=1)
        vocab_norm = F.normalize(vocab_bank_tensor, p=2, dim=1)
        sim_matrix = user_norm @ vocab_norm.T

        top_scores, top_indices = torch.topk(sim_matrix, k=3, dim=1)
        predicted_labels = torch.argmax(logits, dim=1).tolist()

        results = []
        for i, item in enumerate(items):
            recs = [
                {"word": vocab_bank_words[idx], "similarity": round(score, 4)}
                for score, idx in zip(top_scores[i].tolist(), top_indices[i].tolist())
            ]
            entry = {"input": item, "top_3_recommendations": recs}
            if debug:
                entry["predicted_error_type"] = idx_to_class.get(str(predicted_labels[i]), str(predicted_labels[i]))
            results.append(entry)

        return func.HttpResponse(
            json.dumps({"results": results}),
            mimetype="application/json",
            status_code=200
        )

    except ValueError:
        return func.HttpResponse("Invalid JSON input.", status_code=400)
    except Exception as e:
        logging.error(f"Inference error: {str(e)}")
        return func.HttpResponse(f"Server Error: {str(e)}", status_code=500)