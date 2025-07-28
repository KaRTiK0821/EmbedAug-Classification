import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from model import CryClassifier
from dataset import BabyCryDataset
from train import train
from evaluate import evaluate
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchaudio
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn.functional as F
import random
from torch.utils.data import Subset


TARGET_LENGTH = 64

class PadCropTransform:
    def __init__(self, target_length=64, augment=True):
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=64
        )
        self.target_length = target_length
        self.augment = augment

    def add_noise(self, waveform, noise_level=0.005):
        return waveform + noise_level * torch.randn_like(waveform)

    def time_shift(self, waveform, shift_limit=0.02):
        shift_amt = int(random.uniform(-shift_limit, shift_limit) * waveform.shape[-1])
        return torch.roll(waveform, shifts=shift_amt, dims=-1)

    def volume_scale(self, waveform, scale_range=(0.95, 1.05)):
        scale = random.uniform(*scale_range)
        return waveform * scale

    def __call__(self, waveform):
        if self.augment and self.training:
            waveform = self.add_noise(waveform)
            waveform = self.time_shift(waveform)
            waveform = self.volume_scale(waveform)

        mel = self.melspec(waveform)

        if mel.shape[-1] > self.target_length:
            mel = mel[:, :, :self.target_length]
        elif mel.shape[-1] < self.target_length:
            pad_len = self.target_length - mel.shape[-1]
            mel = torch.nn.functional.pad(mel, (0, pad_len))

        return mel
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_transform = PadCropTransform(target_length=64, augment=True)
train_transform.training = True  # augmentations ON

eval_transform = PadCropTransform(target_length=64, augment=False)
eval_transform.training = False  # no augmentations


full_dataset = BabyCryDataset("data", transform=None)
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_indices, val_indices, test_indices = torch.utils.data.random_split(
    range(total_size),
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_dataset = Subset(BabyCryDataset("data", transform=train_transform), train_indices)
val_dataset = Subset(BabyCryDataset("data", transform=eval_transform), val_indices)
test_dataset = Subset(BabyCryDataset("data", transform=eval_transform), test_indices)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

model = CryClassifier(n_classes=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# Count class instances
targets = [label for _, label in full_dataset.samples]
class_counts = Counter(targets)
print("🔢 Class counts:", class_counts)

# Compute weights
total = sum(class_counts.values())
weights = [total / class_counts[i] for i in range(len(class_counts))]
norm_weights = torch.tensor(weights, dtype=torch.float)
norm_weights = norm_weights / norm_weights.sum()
print("⚖️ Normalized class weights:", norm_weights.tolist())

criterion = torch.nn.CrossEntropyLoss(weight=norm_weights.to(device))
train_losses, val_losses = train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs=20,
    patience=3,
    save_path="outputs/best_model.pth"
)


# Plot and save loss curve
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss", marker='o')
plt.plot(val_losses, label="Val Loss", marker='s')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/train_val_loss_curve.png")
# plt.show()


print("\n📊 Validation Set Performance:")
evaluate(model, val_loader, device, full_dataset.classes)

print("\n🧪 Final Test Set Performance:")
evaluate(model, test_loader, device, full_dataset.classes, export_csv=True, split_name="test")

