import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class SushiClassifier(nn.Module):
    def __init__(self, num_species, num_parts):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.base_model = models.resnet18(weights=weights)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # remove the original FC

        self.species_head = nn.Linear(in_features, num_species)
        self.part_head = nn.Linear(in_features, num_parts)

    def forward(self, x):
        features = self.base_model(x)
        species_logits = self.species_head(features)
        part_logits = self.part_head(features)
        return species_logits, part_logits

def train(model, dataloader, species_criterion, part_criterion, optimizer, device):
    model.train()
    for images, species_labels, part_labels in dataloader:
        images = images.to(device)
        species_labels = species_labels.to(device)
        part_labels = part_labels.to(device)

        optimizer.zero_grad()
        species_logits, part_logits = model(images)

        loss_species = species_criterion(species_logits, species_labels)
        loss_part = part_criterion(part_logits, part_labels)
        loss = loss_species + loss_part

        loss.backward()
        optimizer.step()
