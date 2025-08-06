import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class FishClassifier(nn.Module):
    def __init__(self, species_list):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.species_classifier = models.resnet18(weights=weights)
        self.species_classifier.fc = nn.Linear(self.species_classifier.fc.in_features, len(species_list))  # remove the original FC
        self.idx_to_species = {i: species for i, species in enumerate(species_list)}

    def forward(self, x):
        return self.species_classifier(x)