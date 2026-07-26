import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Paths
DATA_PATH = '../../data/final_pairs_with_error_labels_WIP_for_INSPECTIONs.xlsx'
MODEL_DIR = '../../models'

# Ensure the models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load the dataset
print("Loading dataset...")
df = pd.read_excel(DATA_PATH)
print("Data loaded successfully. Shape:", df.shape)

# Preprocessing
print("Preprocessing data...")
df['edit_distance'] = df.apply(lambda row: nltk.edit_distance(row['correct_word'], row['misspelled_word']), axis=1)
X = df[['edit_distance']].values  # Features
y = df['error_type'].values  # Target labels

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# Define Dataset class
class SpellingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoaders
train_dataset = SpellingDataset(X_train, y_train)
test_dataset = SpellingDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define Model 1 (BiLSTM)
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Use the last hidden state
        return x

# Define Model 2 (CNN)
class CNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # Change shape for Conv1d
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.fc(x)
        return x

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("Classification Report:\n", classification_report(all_labels, all_preds))

# Train Model 1 (BiLSTM)
print("Training Model 1 (BiLSTM)...")
input_size = X_train.shape[1]
hidden_size = 64
output_size = len(np.unique(y_train))
model1 = BiLSTM(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=0.001)
train_model(model1, train_loader, criterion, optimizer)
evaluate_model(model1, test_loader)

# Save Model 1
torch.save(model1.state_dict(), os.path.join(MODEL_DIR, 'model1_bilstm.pth'))
print("Model 1 saved successfully.")

# Train Model 2 (CNN)
print("Training Model 2 (CNN)...")
model2 = CNN(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model2.parameters(), lr=0.001)
train_model(model2, train_loader, criterion, optimizer)
evaluate_model(model2, test_loader)

# Save Model 2
torch.save(model2.state_dict(), os.path.join(MODEL_DIR, 'model2_cnn.pth'))
print("Model 2 saved successfully.")