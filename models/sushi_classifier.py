import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class SushiClassifier(nn.Module):
    def __init__(self, label_list):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.base_model = models.resnet18(weights=weights)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, len(label_list))
        self.idx_to_type = {i: species for i, species in enumerate(label_list)}

    def forward(self, x):
        return self.base_model(x)