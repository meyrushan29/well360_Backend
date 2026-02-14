"""
Retrain Lip Hydration Model with Improved Preprocessing
This script retrains the model with center cropping to focus on lips
"""

import sys
sys.path.append(r"d:\PP2\Research_Project_225\IT22564818_Meyrushan_N\Model\Human_Body_Hydration_Managment_PP1")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from config import DEVICE, EPOCHS, LR, MODEL_OUT
from dataLoad_images import load_data_images

print("="*60)
print("RETRAINING LIP HYDRATION MODEL")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LR}")
print(f"Model Output: {MODEL_OUT}")
print("="*60)

# Load data with improved preprocessing (center cropping)
print("\nLoading dataset with improved preprocessing...")
train_loader, test_loader, class_names, train_dataset = load_data_images()
print(f"Classes: {class_names}")
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_loader.dataset)}")

# Create model
print("\nInitializing ResNet18 model...")
model = models.resnet18(pretrained=True)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, len(class_names))
)

model.to(DEVICE)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# Training loop
print("\n" + "="*60)
print("TRAINING STARTED")
print("="*60)

best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)

        correct += torch.sum(preds == labels)
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct.double() / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_acc:.4f}"
    )
    
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        print(f"  → New best accuracy: {best_acc:.4f}")

# Evaluation
print("\n" + "="*60)
print("EVALUATION ON TEST SET")
print("="*60)

model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Calculate test accuracy
test_acc = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Save model
torch.save(model.state_dict(), MODEL_OUT)
print(f"\n✅ Model saved successfully → {MODEL_OUT}")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Best Training Accuracy: {best_acc:.4f}")
print(f"Final Test Accuracy: {test_acc:.4f}")
print("\nNext steps:")
print("1. Test the model with new images")
print("2. Check XAI heatmaps to verify focus on lips")
print("3. Deploy to backend if performance is good")
print("="*60)
