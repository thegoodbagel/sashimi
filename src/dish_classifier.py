import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class DishClassifier(nn.Module):
    def __init__(self, dish_list):
        super().__init__()
        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, len(dish_list))
        self.idx_to_dish = {i: dish for i, dish in enumerate(dish_list)}

    def forward(self, x):
        return self.backbone(x)
