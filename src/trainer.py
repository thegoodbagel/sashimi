import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from sushi_classifier import SushiClassifier, train
from data import SushiDataset
from sklearn.model_selection import train_test_split

def main():
    # Paths
    label_path = "sushi_labels.csv"
    image_dir = "raw_images"
    model_save_path = "saved_models/best_model.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Load labels
    df = pd.read_csv(label_path)

    # Encode labels as integers
    label_to_idx = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
    df['label'] = df['label'].map(label_to_idx)

    # Train/val split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    # Transform (ResNet18 expects 224x224 and normalization)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Datasets and loaders
    train_dataset = SushiDataset(train_df, image_dir, transform=transform)
    val_dataset = SushiDataset(val_df, image_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Model, loss, optimizer
    num_classes = len(label_to_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SushiClassifier(num_classes=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0

    # Train loop
    for epoch in range(10):
        train(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}: Val Accuracy = {val_acc:.2f}")

        if val_acc > best_val_acc:
            print("✅ New best model! Saving...")
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

if __name__ == "__main__":
    main()
