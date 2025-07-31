import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F

class SushiClassifier(nn.Module):
    def __init__(self, species_list):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.species_classifier = models.resnet18(weights=weights)
        self.species_classifier.fc = nn.Linear(self.species_classifier.fc.in_features, len(species_list))  # remove the original FC
        self.idx_to_species = {i: species for i, species in enumerate(species_list)}

    def forward(self, x):
        return self.base_model(x)

def predict(model, input_image, transform, device):
    model.eval()

    # Preprocess and batch the image
    image_tensor = transform(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        species_logits = model(image_tensor)
        probs = F.softmax(species_logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    label = model.idx_to_species[pred_idx.item()]
    return label, confidence.item()