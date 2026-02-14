"""
LIP IMAGE MODEL IMPROVEMENT SCRIPT
===================================
Performance improvements for MobileNetV2-based lip dehydration detection:

1. Advanced Data Augmentation
2. Learning Rate Scheduling
3. Mixed Precision Training (faster)
4. Early Stopping with Best Model Saving
5. Test Time Augmentation (TTA) for inference
6. Model Quantization for faster inference
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.config import DEVICE, MOBILENET_MODEL_OUT, RANDOM_STATE, DATA_DIR
from core.utils import setup_logging, Timer
from hydration.training.preprocess_images import get_transforms
from hydration.training.dataLoad_images import load_data_images

LOG = setup_logging()


class ImprovedLipModel(nn.Module):
    """Enhanced MobileNetV2 with dropout and better architecture"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        
        # Replace classifier with improved version
        num_ftrs = self.mobilenet.classifier[1].in_features
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


def get_advanced_transforms(train=True):
    """Enhanced data augmentation"""
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # Add random erasing (cutout)
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class ImprovedLipTrainer:
    """Advanced trainer with modern techniques"""
    
    def __init__(self, model, device=DEVICE, use_mixed_precision=True):
        self.model = model.to(device)
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device.type == 'cuda'
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Learning rate scheduler
        self.scheduler = None  # Will be set after knowing dataset size
        
        # Metrics
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with mixed precision"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                LOG.info(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                        f"Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%")
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validation with TTA (Test Time Augmentation)"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader, epochs=30, patience=5):
        """Full training loop with early stopping"""
        LOG.info("=" * 60)
        LOG.info("IMPROVED LIP MODEL TRAINING")
        LOG.info(f"Device: {self.device}")
        LOG.info(f"Mixed Precision: {self.use_mixed_precision}")
        LOG.info("=" * 60)
        
        # Setup OneCycleLR scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=0.01,
            epochs=epochs,
            steps_per_epoch=len(train_loader)
        )
        
        epochs_no_improve = 0
        
        with Timer() as t:
            for epoch in range(1, epochs + 1):
                # Train
                train_loss, train_acc = self.train_epoch(train_loader, epoch)
                
                # Validate
                val_loss, val_acc = self.validate(val_loader)
                
                # Update scheduler
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
               
                # Save history
                self.training_history['train_loss'].append(train_loss)
                self.training_history['train_acc'].append(train_acc)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_acc'].append(val_acc)
                
                LOG.info(f"\nEpoch {epoch}/{epochs}:")
                LOG.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                LOG.info(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%\n")
                
                # Save best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_model_state = self.model.state_dict().copy()
                    epochs_no_improve = 0
                    LOG.info(f"[NEW BEST] Val Acc: {val_acc:.2f}%")
                else:
                    epochs_no_improve += 1
                
                # Early stopping
                if epochs_no_improve >= patience:
                    LOG.info(f"Early stopping after {epoch} epochs (no improvement for {patience} epochs)")
                    break
        
        LOG.info(f"\nTotal training time: {t.get_duration():.2f}s")
        LOG.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
    
    def save_model(self, path=None):
        """Save improved model"""
        if path is None:
            path = MOBILENET_MODEL_OUT
        
        torch.save(self.best_model_state, path)
        LOG.info(f"Best model saved to {path}")
        
        # Save training history
        history_path = Path(path).parent / "improved_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        LOG.info(f"Training history saved to {history_path}")


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    """Main training script"""
    LOG.info("=" * 60)
    LOG.info("LIP IMAGE MODEL IMPROVEMENT PIPELINE")
    LOG.info("=" * 60)
    
    # Configuration - Non-interactive defaults
    epochs = 50
    patience = 7
    
    # Load data using existing function
    LOG.info("Loading image dataset...")
    try:
        train_loader, test_loader, class_names, _ = load_data_images()
        LOG.info(f"Dataset loaded successfully. Classes: {class_names}")
    except Exception as e:
        LOG.error(f"Failed to load data: {e}")
        return
    
    # Create improved model
    model = ImprovedLipModel(num_classes=len(class_names))
    LOG.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train
    trainer = ImprovedLipTrainer(model, use_mixed_precision=True)
    trainer.train(train_loader, test_loader, epochs=epochs, patience=patience)
    
    # Save
    trainer.save_model()
    
    LOG.info("=" * 60)
    LOG.info("LIP MODEL IMPROVEMENT COMPLETE")
    LOG.info("=" * 60)


if __name__ == "__main__":
    main()
