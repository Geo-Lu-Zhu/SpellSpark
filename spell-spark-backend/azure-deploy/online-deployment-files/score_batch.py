import os
import json
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import tokenizer
import glob

from azureml.ai.monitoring import Collector
from azureml.ai.monitoring.context import BasicCorrelationContext

# =====================================================================
# 1. MODEL ARCHITECTURE (same as score.py)
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
# 2. INITIALIZATION
# =====================================================================
def init():
    global model, config, vocab_bank_tensor, vocab_bank_words
    global inputs_collector, outputs_collector

    model_root = os.getenv('AZUREML_MODEL_DIR')

    config_path = glob.glob(os.path.join(model_root, '**', 'model1_config.json'), recursive=True)[0]
    weights_path = glob.glob(os.path.join(model_root, '**', 'model1_bilstm.pth'), recursive=True)[0]

    with open(config_path, 'r') as f:
        config = json.load(f)

    model = Model1_BiLSTM(vocab_size=config['vocab_size'], num_classes=config['num_classes'])
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    current_dir = os.path.dirname(__file__)
    csv_path = glob.glob(os.path.join(current_dir, '*vocab_profiles*.csv'))[0]
    vocab_df = pd.read_csv(csv_path)

    vocab_bank_words = vocab_df['answer'].tolist()
    profiles_list = [json.loads(prof) for prof in vocab_df['predicted_profiles']]
    vocab_bank_tensor = torch.tensor(profiles_list, dtype=torch.float32)

    inputs_collector = Collector(name='model_inputs')
    outputs_collector = Collector(name='model_outputs')

# =====================================================================
# 3. INFERENCE
# =====================================================================
def run(raw_data):
    try:
        # Input schema:
        # {"requests": [{"correct_word": "...", "incorrect_word": "..."}, ...], "debug": false}
        request = json.loads(raw_data)
        items = request.get('requests', [])
        debug = bool(request.get('debug', False))

        if not items:
            return {"error": "Payload must contain 'requests': a non-empty list of {correct_word, incorrect_word} objects."}

        char2idx = config.get('char2idx', None)
        idx_to_class = config.get('idx_to_class', {})
        max_seq_len = config.get('max_seq_len', 55)

        # Log inputs
        input_df = pd.DataFrame(items)
        context = BasicCorrelationContext(id='spellspark_request')
        context = inputs_collector.collect(input_df, context)

        seq_tensors = []
        for item in items:
            seq_ids = tokenizer.encode_pair(
                str(item.get('correct_word', '')),
                str(item.get('incorrect_word', '')),
                char2idx=char2idx
            )
            seq_tensors.append(seq_ids)

        # fixed-length padding matches training collate_fn exactly
        padded = [(s + [0] * max_seq_len)[:max_seq_len] for s in seq_tensors]
        input_tensor = torch.tensor(padded, dtype=torch.long)

        with torch.no_grad():
            logits, user_embeddings = model(input_tensor)  # [B, num_classes], [B, 64]

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
            entry = {
                "input": item,
                "top_3_recommendations": recs
            }
            if debug:
                entry["predicted_error_type"] = idx_to_class.get(str(predicted_labels[i]), str(predicted_labels[i]))
            results.append(entry)

        # Log outputs
        output_df = pd.DataFrame([{"recommendations": [r["top_3_recommendations"][0]["word"] for r in results]}])
        outputs_collector.collect(output_df, context)

        return {"results": results}

    except Exception as e:
        return {"error": str(e)}
