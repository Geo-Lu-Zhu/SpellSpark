from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

try:
    from .char_tokenizer import CharTokenizer, PAD_IDX, build_tokenizer_from_pairs
except ImportError:
    from char_tokenizer import CharTokenizer, PAD_IDX, build_tokenizer_from_pairs


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]
DATA_CANDIDATES = [
    REPO_ROOT / "data" / "final_pairs_with_error_labels_WIP_for_INSPECTIONs.xlsx",
    BACKEND_ROOT / "data" / "final_pairs_with_error_labels_WIP_for_INSPECTIONs.xlsx",
]
MODEL_DIR = BACKEND_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def resolve_data_path() -> Path:
    """Return the first existing dataset path."""
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find final_pairs_with_error_labels_WIP_for_INSPECTIONs.xlsx in the repo."
    )


def load_dataset() -> pd.DataFrame:
    """Load the Excel dataset into a DataFrame."""
    data_path = resolve_data_path()
    print(f"Loading dataset from {data_path}...")
    if data_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)
    print("Data loaded successfully. Shape:", df.shape)
    return df


def validate_required_columns(df: pd.DataFrame) -> None:
    """Ensure the expected input columns are present."""
    required_columns = ("correct", "incorrect", "error_type")
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")


def collate_sequences(batch):
    """Pad a batch of variable-length token sequences."""
    sequences, labels = zip(*batch)
    max_len = max(len(sequence) for sequence in sequences)
    padded = [sequence + [PAD_IDX] * (max_len - len(sequence)) for sequence in sequences]
    return torch.tensor(padded, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class PairSequenceDataset(Dataset):
    def __init__(self, pairs, labels, tokenizer: CharTokenizer):
        """Store tokenized word pairs and labels."""
        self.sequences = [tokenizer.encode_pair(correct, incorrect) for correct, incorrect in pairs]
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class Model1_BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, num_layers=2, num_classes=5, dropout=0.3):
        """Build the BiLSTM classifier used for Model 1."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """Run a forward pass through the BiLSTM."""
        emb = self.dropout(self.embedding(x))
        _, (h_n, _) = self.lstm(emb)
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h = torch.cat([h_fwd, h_bwd], dim=1)
        h = self.dropout(h)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


class Model2_CharCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_filters=64, kernel_sizes=(2, 3, 4), num_classes=5, dropout=0.5):
        """Build the character CNN used for Model 2."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=kernel) for kernel in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        total_filters = num_filters * len(kernel_sizes)
        self.fc1 = nn.Linear(total_filters, total_filters // 2)
        self.fc2 = nn.Linear(total_filters // 2, num_classes)

    def forward(self, x):
        """Run a forward pass through the Char-CNN."""
        emb = self.embedding(x).permute(0, 2, 1)
        pooled = []
        for conv in self.convs:
            out = F.relu(conv(emb))
            pooled.append(out.max(dim=2).values)
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        h = F.relu(self.fc1(cat))
        return self.fc2(h)


def train_epoch(model, loader, optimizer, device):
    """Train one epoch and return loss and accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(x_batch)
        loss = F.cross_entropy(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate a model and return loss and accuracy."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        loss = F.cross_entropy(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def get_predictions(model, loader, device):
    """Collect labels and predictions for a loader."""
    model.eval()
    all_preds = []
    all_labels = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        logits = model(x_batch)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

    return all_labels, all_preds


def main():
    """Train both models and save the artifacts."""
    print("Loading and preprocessing data...")
    df = load_dataset()

    validate_required_columns(df)

    df = df[["correct", "incorrect", "error_type"]].dropna().copy()
    df["correct"] = df["correct"].astype(str)
    df["incorrect"] = df["incorrect"].astype(str)
    df["error_type"] = df["error_type"].astype(str)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["error_type"])
    pairs = list(zip(df["correct"], df["incorrect"]))

    tokenizer = build_tokenizer_from_pairs(pairs)
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

    train_pairs, val_pairs, train_labels, val_labels = train_test_split(
        pairs,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    train_dataset = PairSequenceDataset(train_pairs, train_labels, tokenizer)
    val_dataset = PairSequenceDataset(val_pairs, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_sequences)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_sequences)

    num_classes = len(label_encoder.classes_)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 30

    print("Training Model 1 (BiLSTM)...")
    model1 = Model1_BiLSTM(vocab_size=tokenizer.vocab_size, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model1.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_epoch(model1, train_loader, optimizer, device)
        vl_loss, vl_acc = evaluate(model1, val_loader, device)
        scheduler.step(vl_loss)
        print(f"Epoch {epoch:3d} | train {tr_loss:.4f} / {tr_acc:.3f} | val {vl_loss:.4f} / {vl_acc:.3f}")

    true_labels, pred_labels = get_predictions(model1, val_loader, device)
    print(classification_report(true_labels, pred_labels, target_names=label_encoder.classes_))
    print("Validation accuracy:", accuracy_score(true_labels, pred_labels))
    torch.save(model1.state_dict(), MODEL_DIR / "model1_bilstm.pth")

    print("Training Model 2 (Char-CNN)...")
    model2 = Model2_CharCNN(vocab_size=tokenizer.vocab_size, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model2.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_epoch(model2, train_loader, optimizer, device)
        vl_loss, vl_acc = evaluate(model2, val_loader, device)
        scheduler.step(vl_loss)
        print(f"Epoch {epoch:3d} | train {tr_loss:.4f} / {tr_acc:.3f} | val {vl_loss:.4f} / {vl_acc:.3f}")

    true_labels, pred_labels = get_predictions(model2, val_loader, device)
    print(classification_report(true_labels, pred_labels, target_names=label_encoder.classes_))
    print("Validation accuracy:", accuracy_score(true_labels, pred_labels))
    torch.save(model2.state_dict(), MODEL_DIR / "model2_cnn.pth")

    joblib.dump(tokenizer, MODEL_DIR / "char_tokenizer.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")


if __name__ == "__main__":
    main()