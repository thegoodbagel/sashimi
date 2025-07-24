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