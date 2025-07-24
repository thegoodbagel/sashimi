import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from sushi_classifier import SushiClassifier, train
from sushi_dataset import SushiDataset
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

def main():
    # Paths
    label_path = "./data/sushi_labels.csv"
    image_dir = "./data/processed"
    model_save_path = "saved_models/best_model.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Load and prepare dataframe
    df = pd.read_csv(label_path)
    df = df.dropna(subset=["species", "part"])  # Ensure no missing labels

    # Create label encodings
    species_to_idx = {label: idx for idx, label in enumerate(sorted(df["species"].unique()))}
    part_to_idx = {label: idx for idx, label in enumerate(sorted(df["part"].unique()))}

    df["species"] = df["species"].map(species_to_idx)
    df["part"] = df["part"].map(part_to_idx)

    # Train/val split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["species"], random_state=42)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Datasets and loaders
    train_dataset = SushiDataset(train_df, image_dir, transform=transform,
                                 species_to_idx=species_to_idx, part_to_idx=part_to_idx)
    val_dataset = SushiDataset(val_df, image_dir, transform=transform,
                               species_to_idx=species_to_idx, part_to_idx=part_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Model, losses, optimizer
    num_species = len(species_to_idx)
    num_parts = len(part_to_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SushiClassifier(num_species=num_species, num_parts=num_parts).to(device)

    species_criterion = CrossEntropyLoss()
    part_criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0

    # Training loop
    for epoch in range(10):
        train(model, train_loader, species_criterion, part_criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}: Val Accuracy = {val_acc:.2f}")

        if val_acc > best_val_acc:
            print("✅ New best model! Saving...")
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)

def evaluate(model, dataloader, device):
    model.eval()
    correct_species = 0
    correct_parts = 0
    total = 0

    with torch.no_grad():
        for images, species_labels, part_labels in dataloader:
            images = images.to(device)
            species_labels = species_labels.to(device)
            part_labels = part_labels.to(device)

            species_logits, part_logits = model(images)

            _, pred_species = torch.max(species_logits, 1)
            _, pred_parts = torch.max(part_logits, 1)

            correct_species += (pred_species == species_labels).sum().item()
            correct_parts += (pred_parts == part_labels).sum().item()
            total += species_labels.size(0)

    species_acc = correct_species / total
    part_acc = correct_parts / total

    print(f"🔍 Species Accuracy: {species_acc:.2f} | Part Accuracy: {part_acc:.2f}")
    return (species_acc + part_acc) / 2

if __name__ == "__main__":
    main()
