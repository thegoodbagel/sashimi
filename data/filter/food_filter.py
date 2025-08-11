import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import numpy as np


SUSHI_DIR = './data/filter/sushi_examples'
NONSUSHI_DIR = './data/filter/non_sushi_examples'
BEST_MODEL_PATH = './data/filter/best_sushi_filter.pth'

# --- Dataset ---
class SushiFilterDataset(Dataset):
    def __init__(self, sushi_dir, nonsushi_dir, transform=None):
        self.sushi_images = [os.path.join(sushi_dir, f) for f in os.listdir(sushi_dir) if f.lower().endswith(('jpg','png','jpeg'))]
        self.nonsushi_images = [os.path.join(nonsushi_dir, f) for f in os.listdir(nonsushi_dir) if f.lower().endswith(('jpg','png','jpeg'))]
        self.transform = transform

        # Balance dataset by using min count
        self.length = min(len(self.sushi_images), len(self.nonsushi_images))

    def __len__(self):
        return self.length * 2  # sushi + nonsushi balanced

    def __getitem__(self, idx):
        if idx < self.length:
            img_path = self.sushi_images[idx]
            label = 1  # sushi/sashimi
        else:
            img_path = self.nonsushi_images[idx - self.length]
            label = 0  # not sushi

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label

# --- Model ---
class SushiFilterModel(nn.Module):
    def __init__(self):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT
        self.base_model = mobilenet_v2(weights=weights)
        self.base_model.classifier[1] = nn.Linear(self.base_model.last_channel, 2)  # binary output

    def forward(self, x):
        return self.base_model(x)

# --- Training ---
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# --- Evaluation ---
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# --- Main ---
def main():
    # Transform for MobileNetV2
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = SushiFilterDataset(SUSHI_DIR, NONSUSHI_DIR, transform=transform)

    # Get indices of sushi and nonsushi
    sushi_indices = list(range(dataset.length))
    nonsushi_indices = list(range(dataset.length, dataset.length * 2))

    # Shuffle
    np.random.shuffle(sushi_indices)
    np.random.shuffle(nonsushi_indices)

    # Split each class 80/20
    sushi_train_len = int(0.8 * len(sushi_indices))
    nonsushi_train_len = int(0.8 * len(nonsushi_indices))

    train_indices = sushi_indices[:sushi_train_len] + nonsushi_indices[:nonsushi_train_len]
    val_indices   = sushi_indices[sushi_train_len:] + nonsushi_indices[nonsushi_train_len:]

    # Shuffle final sets (so sushi/nonsushi are mixed)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SushiFilterModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10
    best_acc = 0
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("Saved best model!")

# --- Inference helper ---
def predict(model, image, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
        return pred.item(), confidence.item()

if __name__ == "__main__":
    main()
