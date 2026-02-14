import os
import torch
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import get_dataloader
from other import EmotionCNN
from config import DATA_DIR, DEVICE, MODEL_PATH, EMOTION_CLASSES

# -------------------- TEST TRANSFORM --------------------
test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def evaluate():
    # Load test data (⚠️ transform MUST be passed)
    test_loader, _ = get_dataloader(
        os.path.join(DATA_DIR, "test"),
        transform=test_transform,
        shuffle=False,
        return_labels=True
    )

    # Load model
    model = EmotionCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n📊 Classification Report:\n")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=EMOTION_CLASSES,
            digits=4
        )
    )

    print("📌 Confusion Matrix:\n")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    evaluate()
