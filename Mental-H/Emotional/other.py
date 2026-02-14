

import torch.nn as nn
from torchvision import models
from config import EMOTION_CLASSES

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=len(EMOTION_CLASSES)):
        super(EmotionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128*6*6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ResNetEmotion(nn.Module):
    def __init__(self, num_classes=len(EMOTION_CLASSES)):
        super(ResNetEmotion, self).__init__()
        # Load ResNet18 (pretrained optional, but recommended)
        # Using weights=None to avoid download if internet issues, but assuming pretrained makes sense if we trained it
        # Since we are loading weights anyway, pretrained=True/False doesn't matter much for structure
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify the first layer to accept grayscale (1 channel) inputs if needed?
        # NO, ResNet expects 3 channels usually. If we trained on Grayscale, we might have modified conv1.
        # But 'best_resnet_model.pth' keys showed 'model.conv1.weight'.
        # Let's assume standard RGB input for now.
        
        # Modify the final Fully Connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
