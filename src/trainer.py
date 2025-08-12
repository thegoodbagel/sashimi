import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from sushi_classifier import SushiClassifier
from sushi_dataset import SushiDataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sklearn.model_selection import StratifiedShuffleSplit
import os


def main():
    # Paths
    label_path = os.getenv("LABEL_PATH", "./data/fish/sushi_labels.csv")
    image_dir = os.getenv("IMAGE_DIR", "./data/fish/processed")
    model_save_path = os.getenv("MODEL_SAVE_PATH", "saved_models/fish/best_model.pth")
    target_column = os.getenv("TARGET_COLUMN", "type")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    print("Label path:", label_path)
    print("Image directory:", image_dir)
    print("Model save path:", model_save_path)

    # Load and prepare dataframe
    df = pd.read_csv(label_path)
    df = df.dropna(subset=[target_column])

    # Remove rare type classes (less than 6 instances)
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
    train_dataset = SushiDataset(train_df, image_dir, target_column, transform=transform)
    val_dataset = SushiDataset(val_df, image_dir, target_column, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model, losses, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SushiClassifier(label_list).to(device)

    type_criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0

    # Training loop
    for epoch in range(10):
        train(model, train_loader, type_criterion, optimizer, device)
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
    for images, type_labels in dataloader:
        images = images.to(device)
        type_labels = type_labels.to(device)

        logits, loss = _forward_pass(model, images, type_labels, loss_function)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, dataloader, device, label_to_idx):
    model.eval()

    num_classes = len(label_to_idx)
    correct_per_class = [0] * num_classes
    total_per_class = [0] * num_classes

    with torch.no_grad():
        for images, type_labels in dataloader:
            images = images.to(device)
            type_labels = type_labels.to(device)

            type_logits = model(images)
            _, pred_type = torch.max(type_logits, 1)

            for label, pred in zip(type_labels, pred_type):
                label = label.item()
                pred = pred.item()
                total_per_class[label] += 1
                if label == pred:
                    correct_per_class[label] += 1

    # Calculate accuracy per class
    acc_per_class = []
    for i in range(num_classes):
        if total_per_class[i] > 0:
            acc = correct_per_class[i] / total_per_class[i]
        else:
            acc = 0.0
        acc_per_class.append(acc)

    # Overall accuracy
    overall_acc = sum(correct_per_class) / sum(total_per_class)

    # Map idx to label name for printing
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    print("\nPer-class accuracy:")
    for i, acc in enumerate(acc_per_class):
        print(f"  Class '{idx_to_label[i]}': {acc:.4f}")

    return overall_acc



if __name__ == "__main__":
    main()
