import torch.nn as nn
import torchvision.models as models

class SushiClassifier(nn.Module):
    def __init__(self, num_species, num_parts):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # remove the original FC

        self.species_head = nn.Linear(in_features, num_species)
        self.part_head = nn.Linear(in_features, num_parts)

    def forward(self, x):
        features = self.base_model(x)
        species_logits = self.species_head(features)
        part_logits = self.part_head(features)
        return species_logits, part_logits
