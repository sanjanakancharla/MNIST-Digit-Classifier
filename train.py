"""
MNIST digit classifier - a small PyTorch CNN.

Trains on the MNIST handwritten-digit dataset (70,000 grayscale images, 28x28),
then reports test accuracy. Expect ~99% after 5 epochs.

Run:  pip install -r requirements.txt
      python train.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Normalize with MNIST's mean/std so training is stable
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root="data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1000)


class DigitCNN(nn.Module):
    """Two conv blocks + a classifier head."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 28x28 -> 26x26 (conv) -> 13x13 (pool)
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 13x13 -> 11x11 (conv) -> 5x5 (pool)
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),  # 10 digit classes
        )

    def forward(self, x):
        return self.net(x)


model = DigitCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


def train_epoch() -> float:
    model.train()
    correct, total = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total += labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    return correct / total


@torch.no_grad()
def evaluate() -> float:
    model.eval()
    correct = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(dim=1) == labels).sum().item()
    return correct / len(test_ds)


EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    train_acc = train_epoch()
    test_acc = evaluate()
    print(f"Epoch {epoch}/{EPOCHS}  train acc: {train_acc:.4f}  test acc: {test_acc:.4f}")

torch.save(model.state_dict(), "mnist_cnn.pt")
print("Done - saved model weights to mnist_cnn.pt")
