import torch.nn as nn
from embed_aug_layer import EmbedAug

class CryClassifier(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.embedaug = EmbedAug(mask_prob=0.1, mode='gaussian')
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),  # adjust if input size changes
            nn.ReLU(),
            nn.Dropout(0.3),  # new dropout layer
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.embedaug(x)
        out = self.fc(x)
        return out
