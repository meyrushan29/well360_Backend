"""
COMPREHENSIVE LIP HYDRATION MODEL TRAINING SCRIPT
==================================================

This script trains a MobileNetV2 model for lip-based hydration detection.

Requirements:
- Training data in: hydration/data/Dehydrate/ and hydration/data/Normal/
- At least 50 images per class (recommended: 200+ per class)
- Images should be RGB, any size (will be resized to 224x224)

Usage:
    python hydration/training/train_lip_model_complete.py

Output:
    - hydration/models/LipModel_MobileNetV2.pth (trained model)
    - Training metrics and plots
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent.parent
sys.path.append(str(backend_dir))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import json

from core.config import (
    MOBILENET_MODEL_OUT, 
    DEVICE, 
    BATCH_SIZE, 
    EPOCHS, 
    LR, 
    IMG_SIZE,
    RANDOM_STATE
)

from hydration.training.preprocess_images import get_transforms

print("="*60)
print("LIP HYDRATION MODEL TRAINING")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Model Output: {MOBILENET_MODEL_OUT}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning Rate: {LR}")
print("="*60)

# ======================================================
# 1. MODEL ARCHITECTURE
# ======================================================
class ImprovedLipModel(nn.Module):
    """
    Enhanced MobileNetV2 with dropout and batch normalization
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        num_ftrs = self.mobilenet.classifier[1].in_features
        
        # Custom classifier head
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.mobilenet(x)

# ======================================================
# 2. TRAINING FUNCTION
# ======================================================
def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ======================================================
# 3. VALIDATION FUNCTION
# ======================================================
def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ======================================================
# 4. MAIN TRAINING PIPELINE
# ======================================================
def main():
    # Check if training data exists
    data_dir = backend_dir / "hydration" / "data"
    if not data_dir.exists():
        print(f"\n❌ ERROR: Training data directory not found: {data_dir}")
        print("\nPlease create the following directory structure:")
        print("  hydration/data/")
        print("    ├── Dehydrate/  (images of dehydrated lips)")
        print("    └── Normal/     (images of normal/hydrated lips)")
        return
    
    dehydrate_dir = data_dir / "Dehydrate"
    normal_dir = data_dir / "Normal"
    
    if not dehydrate_dir.exists() or not normal_dir.exists():
        print(f"\n❌ ERROR: Training data subdirectories not found!")
        print(f"Expected:")
        print(f"  - {dehydrate_dir}")
        print(f"  - {normal_dir}")
        return
    
    # Count images
    dehydrate_count = len(list(dehydrate_dir.glob("*.[jp][pn]g")))
    normal_count = len(list(normal_dir.glob("*.[jp][pn]g")))
    
    print(f"\n📊 Training Data:")
    print(f"  Dehydrate: {dehydrate_count} images")
    print(f"  Normal: {normal_count} images")
    print(f"  Total: {dehydrate_count + normal_count} images")
    
    if dehydrate_count < 20 or normal_count < 20:
        print(f"\n⚠️  WARNING: Insufficient training data!")
        print(f"  Minimum recommended: 50 images per class")
        print(f"  For best results: 200+ images per class")
        response = input("\n  Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Load dataset
    print("\n📁 Loading dataset...")
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    full_dataset = datasets.ImageFolder(str(data_dir), transform=train_transform)
    
    # Split dataset (80% train, 20% validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_STATE)
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"✅ Dataset loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Classes: {full_dataset.classes}")
    
    # Initialize model
    print(f"\n🔧 Initializing model...")
    model = ImprovedLipModel(num_classes=len(full_dataset.classes))
    model = model.to(DEVICE)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Learning rate scheduler (without verbose for PyTorch compatibility)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3
    )
    print("✅ Learning rate scheduler initialized")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # Training loop
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    print("="*60)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, DEVICE)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  ✅ New best model! (Val Acc: {val_acc:.2f}%)")
            
            # Save model
            os.makedirs(MOBILENET_MODEL_OUT.parent, exist_ok=True)
            torch.save(model.state_dict(), MOBILENET_MODEL_OUT)
            print(f"  💾 Model saved to: {MOBILENET_MODEL_OUT}")
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {MOBILENET_MODEL_OUT}")
    
    # Save training history
    history_file = MOBILENET_MODEL_OUT.parent / "training_history.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_file}")
    
    # Plot training curves
    plot_training_curves(history)
    
    return model, history

# ======================================================
# 5. PLOTTING FUNCTION
# ======================================================
def plot_training_curves(history):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = MOBILENET_MODEL_OUT.parent / "training_curves.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📈 Training curves saved to: {plot_file}")
    
    plt.close()

# ======================================================
# 6. RUN TRAINING
# ======================================================
if __name__ == "__main__":
    try:
        model, history = main()
        print("\n✅ Script completed successfully!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR during training:")
        print(f"{str(e)}")
        import traceback
        traceback.print_exc()
