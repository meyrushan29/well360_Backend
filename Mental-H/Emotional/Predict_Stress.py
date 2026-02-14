import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# -------------------------------
print("Predict Stress Script started.")
# CONFIG
# -------------------------------
SEQ_LENGTH = 6
BATCH_SIZE = 16
EPOCHS = 30
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIDEO_EMOTION_CSV = "emotion_results.csv"   # video emotion CSV
MODEL_PATH = "stress_lstm.pth"

# Map emotion classes to binary stress label
STRESS_EMOTIONS = ["sad", "angry", "fear", "disgust"]

# -------------------------------
# LOAD VIDEO EMOTION CSV
# -------------------------------
if not os.path.exists(VIDEO_EMOTION_CSV):
    raise FileNotFoundError("emotion_results.csv not found. Run video emotion prediction first.")

print("Loading CSV history...")
df = pd.read_csv(VIDEO_EMOTION_CSV)
print(f"CSV Loaded. Total Samples: {len(df)}")

# Label stress
df["stress"] = df["emotion"].apply(
    lambda x: 1 if str(x).lower() in STRESS_EMOTIONS else 0
)

# Encode emotions
le = LabelEncoder()
df["emotion_encoded"] = le.fit_transform(df["emotion"])
num_classes = len(le.classes_)

# One-hot encode
one_hot = np.eye(num_classes)[df["emotion_encoded"]]

# -------------------------------
# BUILD SEQUENCES
# -------------------------------
X, y = [], []
for i in range(len(one_hot) - SEQ_LENGTH):
    X.append(one_hot[i:i+SEQ_LENGTH])
    y.append(df["stress"].iloc[i+SEQ_LENGTH])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -------------------------------
# DATASET
# -------------------------------
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(
    EmotionDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# -------------------------------
# LSTM MODEL
# -------------------------------
class StressLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.sigmoid(self.fc(out))

model = StressLSTM(num_classes).to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------------
# TRAIN
# -------------------------------
print("\nTraining Stress Model on latest data...\n")

for epoch in range(EPOCHS):
    model.train()
    loss_sum = 0
    for bx, by in train_loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        pred = model(bx)
        loss = criterion(pred, by)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss_sum/len(train_loader):.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print("Stress model updated and saved.")

# -------------------------------
# FINAL VIDEO STRESS PREDICTION (Latest Video)
# -------------------------------
model.eval()

if "video_name" not in df.columns:
    print("Column 'video_name' not found in CSV. Using last rows.")
    target_indices = df.index[-SEQ_LENGTH:]
else:
    last_video = df["video_name"].iloc[-1]
    print(f"\nAnalyzing Stress for Video: {last_video}")
    video_indices = df[df["video_name"] == last_video].index
    
    video_emotions_encoded = df.loc[video_indices, "emotion_encoded"].values
    video_one_hot = np.eye(num_classes)[video_emotions_encoded]
    
    video_probs = []
    
    if len(video_one_hot) < SEQ_LENGTH:
         pad_len = SEQ_LENGTH - len(video_one_hot)
         seq_data = np.pad(video_one_hot, ((pad_len, 0), (0, 0)), 'edge')
         seq = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
         video_probs.append(model(seq).item())
    else:
        for i in range(len(video_one_hot) - SEQ_LENGTH + 1):
            seq_data = video_one_hot[i : i+SEQ_LENGTH]
            seq = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            video_probs.append(model(seq).item())

    if video_probs:
        avg_prob = np.mean(video_probs)
    else:
        avg_prob = 0.0

print("\n==============================")
print("Input Video Stress Analysis")
print("==============================")
print(f"Stress Probability : {avg_prob:.2f}")

if avg_prob >= 0.6:
    print("Result: High Stress Detected")
elif avg_prob >= 0.3:
    print("Result: Moderate Stress Detected")
else:
    print("Result: No Significant Stress")

# -------------------------------
# PLOT STRESS TREND (For Current Video)
# -------------------------------
probs = []
model.eval()
with torch.no_grad():
    if "video_name" in df.columns:
        last_video_name = df["video_name"].iloc[-1]
        video_indices = df[df["video_name"] == last_video_name].index
        v_emotions_encoded = df.loc[video_indices, "emotion_encoded"].values
        v_one_hot = np.eye(num_classes)[v_emotions_encoded]
        
        if len(v_one_hot) >= SEQ_LENGTH:
            for i in range(SEQ_LENGTH, len(v_one_hot) + 1):
                seq = torch.tensor(v_one_hot[i-SEQ_LENGTH:i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                probs.append(model(seq).item())
        else:
            pad_len = SEQ_LENGTH - len(v_one_hot)
            seq_data = np.pad(v_one_hot, ((pad_len, 0), (0, 0)), 'edge')
            seq = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            probs.append(model(seq).item())
    else:
        for i in range(SEQ_LENGTH, len(one_hot)):
            seq = torch.tensor(one_hot[i-SEQ_LENGTH:i], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            probs.append(model(seq).item())

if probs:
    plt.figure(figsize=(10,4))
    plt.plot(probs, marker='o', linestyle='-', color='b', label='Stress Level')
    plt.axhline(0.6, color="red", linestyle="--", label='High Stress Threshold')
    plt.axhline(0.3, color="orange", linestyle="--", label='Moderate Stress Threshold')
    plt.title(f"Stress Trend: {last_video_name if 'video_name' in df.columns else 'All Data'}")
    plt.ylabel("Stress Probability")
    plt.xlabel("Timeline (Sequences)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    print("\nDisplaying Stress Graph. Please close the window to finish...")
    plt.show()
else:
    print("Not enough data to generate stress trend graph.")
