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

RANDOM_STATE = 42
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3


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


class PairSequenceDataset(Dataset):
    """Pair inputs for Model 1: (correct, incorrect) -> hard class label."""

    def __init__(self, pairs, labels, tokenizer: CharTokenizer):
        self.sequences = [tokenizer.encode_pair(correct, incorrect) for correct, incorrect in pairs]
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class WordProfileDataset(Dataset):
    """Word inputs for Model 2: correct_word -> teacher soft class distribution."""

    def __init__(self, words, profiles, tokenizer: CharTokenizer):
        self.sequences = [tokenizer.encode(word) for word in words]
        self.profiles = list(profiles)

    def __len__(self):
        return len(self.profiles)

    def __getitem__(self, idx):
        return self.sequences[idx], self.profiles[idx]


def collate_pair_sequences(batch):
    """Pad tokenized pair sequences and return hard labels."""
    sequences, labels = zip(*batch)
    max_len = max(len(sequence) for sequence in sequences)
    padded = [sequence + [PAD_IDX] * (max_len - len(sequence)) for sequence in sequences]
    return torch.tensor(padded, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_word_profiles(batch):
    """Pad tokenized words and return soft-profile targets."""
    sequences, profiles = zip(*batch)
    max_len = max(len(sequence) for sequence in sequences)
    padded = [sequence + [PAD_IDX] * (max_len - len(sequence)) for sequence in sequences]
    return torch.tensor(padded, dtype=torch.long), torch.tensor(profiles, dtype=torch.float32)


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


def train_classifier_epoch(model, loader, optimizer, device):
    """Train one epoch for hard-label classification (Model 1)."""
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
def evaluate_classifier(model, loader, device):
    """Evaluate one epoch for hard-label classification (Model 1)."""
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
def get_hard_predictions(model, loader, device):
    """Collect hard labels and hard predictions for a loader."""
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


@torch.no_grad()
def generate_teacher_profiles(model, dataset, device):
    """Generate teacher probability distributions for every example in a dataset."""
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pair_sequences)
    all_profiles = []

    for x_batch, _ in loader:
        x_batch = x_batch.to(device)
        probs = F.softmax(model(x_batch), dim=1)
        all_profiles.extend(probs.cpu().tolist())

    return all_profiles


def train_student_epoch(model, loader, optimizer, device):
    """Train one epoch for soft-label distillation (Model 2)."""
    model.train()
    total_loss, total, correct = 0.0, 0, 0
    cosine_sum = 0.0

    for x_batch, target_profiles in loader:
        x_batch = x_batch.to(device)
        target_profiles = target_profiles.to(device)

        logits = model(x_batch)
        log_probs = F.log_softmax(logits, dim=1)
        pred_profiles = log_probs.exp()

        loss = F.kl_div(log_probs, target_profiles, reduction="batchmean")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = target_profiles.size(0)
        total_loss += loss.item() * batch_size
        cosine_sum += F.cosine_similarity(pred_profiles, target_profiles, dim=1).sum().item()

        pred_hard = logits.argmax(dim=1)
        target_hard = target_profiles.argmax(dim=1)
        correct += (pred_hard == target_hard).sum().item()
        total += batch_size

    return total_loss / total, correct / total, cosine_sum / total


@torch.no_grad()
def evaluate_student(model, loader, device):
    """Evaluate one epoch for soft-label distillation (Model 2)."""
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    cosine_sum = 0.0

    for x_batch, target_profiles in loader:
        x_batch = x_batch.to(device)
        target_profiles = target_profiles.to(device)

        logits = model(x_batch)
        log_probs = F.log_softmax(logits, dim=1)
        pred_profiles = log_probs.exp()

        loss = F.kl_div(log_probs, target_profiles, reduction="batchmean")

        batch_size = target_profiles.size(0)
        total_loss += loss.item() * batch_size
        cosine_sum += F.cosine_similarity(pred_profiles, target_profiles, dim=1).sum().item()

        pred_hard = logits.argmax(dim=1)
        target_hard = target_profiles.argmax(dim=1)
        correct += (pred_hard == target_hard).sum().item()
        total += batch_size

    return total_loss / total, correct / total, cosine_sum / total


@torch.no_grad()
def get_student_hard_predictions(model, loader, device):
    """Return argmax(class) targets and predictions for model-2 reporting."""
    model.eval()
    all_pred_hard = []
    all_target_hard = []

    for x_batch, target_profiles in loader:
        x_batch = x_batch.to(device)
        logits = model(x_batch)
        pred_hard = logits.argmax(dim=1).cpu().numpy()
        target_hard = target_profiles.argmax(dim=1).cpu().numpy()
        all_pred_hard.extend(pred_hard)
        all_target_hard.extend(target_hard)

    return all_target_hard, all_pred_hard


def main():
    """Train teacher (model 1), distill soft profiles, then train student (model 2)."""
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
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    train_pair_dataset = PairSequenceDataset(train_pairs, train_labels, tokenizer)
    val_pair_dataset = PairSequenceDataset(val_pairs, val_labels, tokenizer)

    train_pair_loader = DataLoader(
        train_pair_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_pair_sequences,
    )
    val_pair_loader = DataLoader(
        val_pair_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_pair_sequences,
    )

    num_classes = len(label_encoder.classes_)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Training Model 1 (BiLSTM teacher)...")
    model1 = Model1_BiLSTM(vocab_size=tokenizer.vocab_size, num_classes=num_classes).to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=LEARNING_RATE)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, patience=3, factor=0.5)

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = train_classifier_epoch(model1, train_pair_loader, optimizer1, device)
        vl_loss, vl_acc = evaluate_classifier(model1, val_pair_loader, device)
        scheduler1.step(vl_loss)
        print(
            f"Model1 Epoch {epoch:3d} | "
            f"train loss/acc {tr_loss:.4f} / {tr_acc:.3f} | "
            f"val loss/acc {vl_loss:.4f} / {vl_acc:.3f}"
        )

    m1_true_labels, m1_pred_labels = get_hard_predictions(model1, val_pair_loader, device)
    print("\nModel 1 classification report")
    print(classification_report(m1_true_labels, m1_pred_labels, target_names=label_encoder.classes_))
    print("Model 1 validation accuracy:", accuracy_score(m1_true_labels, m1_pred_labels))
    torch.save(model1.state_dict(), MODEL_DIR / "model1_bilstm.pth")

    print("\nGenerating teacher soft profiles for full split dataset...")
    train_soft_profiles = generate_teacher_profiles(model1, train_pair_dataset, device)
    val_soft_profiles = generate_teacher_profiles(model1, val_pair_dataset, device)

    train_words = [correct for correct, _ in train_pairs]
    val_words = [correct for correct, _ in val_pairs]

    train_word_dataset = WordProfileDataset(train_words, train_soft_profiles, tokenizer)
    val_word_dataset = WordProfileDataset(val_words, val_soft_profiles, tokenizer)

    train_word_loader = DataLoader(
        train_word_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_word_profiles,
    )
    val_word_loader = DataLoader(
        val_word_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_word_profiles,
    )

    print("Training Model 2 (Char-CNN student with KL divergence)...")
    model2 = Model2_CharCNN(vocab_size=tokenizer.vocab_size, num_classes=num_classes).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=LEARNING_RATE)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, patience=3, factor=0.5)

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc, tr_cos = train_student_epoch(model2, train_word_loader, optimizer2, device)
        vl_loss, vl_acc, vl_cos = evaluate_student(model2, val_word_loader, device)
        scheduler2.step(vl_loss)
        print(
            f"Model2 Epoch {epoch:3d} | "
            f"train KL/acc/cos {tr_loss:.4f} / {tr_acc:.3f} / {tr_cos:.3f} | "
            f"val KL/acc/cos {vl_loss:.4f} / {vl_acc:.3f} / {vl_cos:.3f}"
        )

    m2_target_hard, m2_pred_hard = get_student_hard_predictions(model2, val_word_loader, device)
    print("\nModel 2 classification report (argmax target profile vs argmax prediction)")
    print(classification_report(m2_target_hard, m2_pred_hard, target_names=label_encoder.classes_))
    print("Model 2 hard validation accuracy:", accuracy_score(m2_target_hard, m2_pred_hard))
    torch.save(model2.state_dict(), MODEL_DIR / "model2_cnn.pth")

    joblib.dump(tokenizer, MODEL_DIR / "char_tokenizer.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")


if __name__ == "__main__":
    main()
