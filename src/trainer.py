import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from fish_classifier import FishClassifier
from fish_dataset import FishDataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sklearn.model_selection import StratifiedShuffleSplit
import os


def main():
    # Paths
    label_path = os.getenv("LABEL_PATH", "./data/dish/sushi_labels.csv")
    image_dir = os.getenv("IMAGE_DIR", "./data/dish/processed")
    model_save_path = os.getenv("MODEL_SAVE_PATH", "saved_models/fish/best_model.pth")
    target_column = os.getenv("TARGET_COLUMN", "type")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    print("Label path:", label_path)
    print("Image directory:", image_dir)
    print("Model save path:", model_save_path)

    # Load and prepare dataframe
    df = pd.read_csv(label_path)
    df = df.dropna(subset=[target_column])

    # Remove rare species classes (less than 6 instances)
    counts = df[target_column].value_counts()
    valid = counts[counts >= 6].index
    df = df[df[target_column].isin(valid)]

    dropped = counts[counts < 6]
    if not dropped.empty:
        print("⚠️ Dropping rare classes with < 6 samples:")
        print(dropped)

    # Create label encodings
    label_list = sorted(df[target_column].unique())
    label_to_idx = {label: idx for idx, label in enumerate(
        sorted(df[target_column].unique()))}
    df[target_column] = df[target_column].map(label_to_idx)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(df, df[target_column]))
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Datasets and loaders
    train_dataset = FishDataset(train_df, image_dir, target_column, transform=transform)
    val_dataset = FishDataset(val_df, image_dir, target_column, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Model, losses, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FishClassifier(label_list).to(device)

    species_criterion = CrossEntropyLoss()
    part_criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0

    # Training loop
    for epoch in range(10):
        train(model, train_loader, species_criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device, label_to_idx)

        print(f"Epoch {epoch+1}: Val Accuracy = {val_acc:.2f}")

        if val_acc > best_val_acc:
            print("✅ New best model! Saving...")
            best_val_acc = val_acc
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'label_list': label_list,
            }
            torch.save(checkpoint, model_save_path)

def _forward_pass(model, images, labels, loss_function):
    logits = model(images)
    loss = loss_function(logits, labels)
    return logits, loss

def train(model, dataloader, loss_function, optimizer, device):
    model.train()
    for images, species_labels in dataloader:
        images = images.to(device)
        species_labels = species_labels.to(device)

        logits, loss = _forward_pass(model, images, species_labels, loss_function)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, dataloader, device, species_to_idx):
    model.eval()
    correct_species = 0
    total = 0

    with torch.no_grad():
        for images, species_labels in dataloader:
            images = images.to(device)
            species_labels = species_labels.to(device)

            species_logits = model(images)
            _, pred_species = torch.max(species_logits, 1)

            correct_species += (pred_species == species_labels).sum().item()
            total += species_labels.size(0)
    species_acc = correct_species / total
    print(f"Accuracy: {species_acc:.2f}")
    return (species_acc) / 2


if __name__ == "__main__":
    main()
