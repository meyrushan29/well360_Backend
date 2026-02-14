import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import time
import copy

from preprocess import get_dataloader
from other import EmotionCNN
from config import DATA_DIR, DEVICE, MODEL_PATH, EMOTION_CLASSES, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS

# ==========================================
# TRAINING CONFIGURATION
# ==========================================
# Enhanced data augmentation for better generalization
train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def train_model():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    print("Loading data...")
    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "test")
    
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found at {train_dir}")
        return

    train_loader = get_dataloader(
        train_dir, 
        transform=train_transform, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    val_loader = None
    if os.path.exists(test_dir):
        val_loader = get_dataloader(
            test_dir, 
            transform=val_transform, 
            batch_size=BATCH_SIZE, 
            shuffle=False
        )
        print("Validation data loaded.")
    
    # 2. Initialize Model
    model = EmotionCNN().to(DEVICE)
    
    # Check if pre-trained model exists to fine-tune
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model weights from {MODEL_PATH} for fine-tuning...")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        except Exception as e:
            print(f"Could not load existing weights (structure might have changed): {e}")
            print("Starting training from scratch.")

    # 3. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # Added weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print("-" * 60)
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training Loop
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
        for imgs, labels in progress_bar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        
        # Validation Loop
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * imgs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            
            print(f"              | Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            scheduler.step(val_acc)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"--> Best model saved! (Acc: {best_acc:.4f})")
        else:
            # If no validation set, just save the latest
            torch.save(model.state_dict(), MODEL_PATH)
            
        print("-" * 60)

    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    
    if val_loader:
        print(f"Best Validation Accuracy: {best_acc:.4f}")
        # Reload best weights
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Final best model saved to {MODEL_PATH}")

if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting...")
