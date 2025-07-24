import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F

class SushiClassifier(nn.Module):
    def __init__(self, num_species, num_parts, idx_to_species):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.base_model = models.resnet18(weights=weights)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # remove the original FC

        self.species_head = nn.Linear(in_features, num_species)
        self.part_head = nn.Linear(in_features, num_parts)
        self.idx_to_species = idx_to_species

    def forward(self, x):
        features = self.base_model(x)
        species_logits = self.species_head(features)
        part_logits = self.part_head(features)
        return species_logits, part_logits

def predict(model, input_image, transform, device):
    model.eval()

    # Preprocess and batch the image
    image_tensor = transform(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        species_logits, _ = model(image_tensor)  # only care about species here
        probs = F.softmax(species_logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    label = model.idx_to_species[pred_idx.item()]
    return label, confidence.item()