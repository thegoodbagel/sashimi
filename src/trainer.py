import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from sushi_classifier import SushiClassifier
from sushi_dataset import SushiDataset
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sklearn.model_selection import StratifiedShuffleSplit

def main():
    # Paths
    label_path = "./data/sushi_labels.csv"
    image_dir = "./data/processed"
    model_save_path = "saved_models/best_model.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Load and prepare dataframe
    df = pd.read_csv(label_path)
    df = df.dropna(subset=["species", "part"])  # Ensure no missing labels

    # Remove rare species classes (less than 6 instances)
    species_counts = df["species"].value_counts()
    valid_species = species_counts[species_counts >= 6].index
    df = df[df["species"].isin(valid_species)]

    dropped_species = species_counts[species_counts < 6]
    if not dropped_species.empty:
        print("⚠️ Dropping rare species classes with < 6 samples:")
        print(dropped_species)


    # Create label encodings
    species_to_idx = {label: idx for idx, label in enumerate(sorted(df["species"].unique()))}
    part_to_idx = {label: idx for idx, label in enumerate(sorted(df["part"].unique()))}

    df["species"] = df["species"].map(species_to_idx)
    df["part"] = df["part"].map(part_to_idx)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(df, df["species"]))
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
    train_dataset = SushiDataset(train_df, image_dir, transform=transform,
                                 species_to_idx=species_to_idx, part_to_idx=part_to_idx)
    val_dataset = SushiDataset(val_df, image_dir, transform=transform,
                               species_to_idx=species_to_idx, part_to_idx=part_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Model, losses, optimizer
    num_species = len(species_to_idx)
    num_parts = len(part_to_idx)
    idx_to_species = {v: k for k, v in species_to_idx.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SushiClassifier(num_species=num_species, num_parts=num_parts, idx_to_species=idx_to_species).to(device)

    species_criterion = CrossEntropyLoss()
    part_criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0

    # Training loop
    for epoch in range(10):
        train(model, train_loader, species_criterion, part_criterion, optimizer, device, species_to_idx)
        val_acc = evaluate(model, val_loader, device, species_to_idx)

        print(f"Epoch {epoch+1}: Val Accuracy = {val_acc:.2f}")

        if val_acc > best_val_acc:
            print("✅ New best model! Saving...")
            best_val_acc = val_acc
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'num_species': num_species,
                'num_parts': num_parts,
                'idx_to_species': model.idx_to_species,
            }
            torch.save(checkpoint, model_save_path)

def _forward_pass(model, images, species_labels, part_labels, species_to_idx, species_criterion, part_criterion):
    species_logits, part_logits = model(images)
    loss_species = species_criterion(species_logits, species_labels)

    maguro_mask = (species_labels == species_to_idx["maguro_(tuna)"])
    if maguro_mask.any():
        loss_part = part_criterion(part_logits[maguro_mask], part_labels[maguro_mask])
        loss = loss_species + loss_part
    else:
        loss = loss_species

    return species_logits, part_logits, loss


def train(model, dataloader, species_criterion, part_criterion, optimizer, device, species_to_idx):
    model.train()
    for images, species_labels, part_labels in dataloader:
        images = images.to(device)
        species_labels = species_labels.to(device)
        part_labels = part_labels.to(device)

        species_logits, part_logits, loss = _forward_pass(
            model, images, species_labels, part_labels,
            species_to_idx, species_criterion, part_criterion
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, dataloader, device, species_to_idx):
    model.eval()
    correct_species = 0
    correct_parts = 0
    total = 0
    part_total = 0
    maguro_idx = species_to_idx["maguro_(tuna)"]

    with torch.no_grad():
        for images, species_labels, part_labels in dataloader:
            images = images.to(device)
            species_labels = species_labels.to(device)
            part_labels = part_labels.to(device)

            species_logits, part_logits = model(images)
            _, pred_species = torch.max(species_logits, 1)
            _, pred_parts = torch.max(part_logits, 1)

            is_maguro = (species_labels == maguro_idx)

            correct_species += (pred_species == species_labels).sum().item()
            total += species_labels.size(0)

            if is_maguro.any():
                correct_parts += (pred_parts[is_maguro] == part_labels[is_maguro]).sum().item()
                part_total += is_maguro.sum().item()

    species_acc = correct_species / total
    part_acc = correct_parts / part_total if part_total > 0 else 0.0

    print(f"🔍 Species Accuracy: {species_acc:.2f} | Part Accuracy (maguro only): {part_acc:.2f}")
    return (species_acc + part_acc) / 2


if __name__ == "__main__":
    main()
