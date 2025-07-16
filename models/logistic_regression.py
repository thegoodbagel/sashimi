import torch
import torch.nn as nn
import torchvision.models as models

class SushiClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SushiClassifier, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

    def train(model, train_loader, criterion, optimizer, device):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    def predict(model, image, transform, class_names, device):
        model.eval()
        with torch.no_grad():
            image = transform(image).unsqueeze(0).to(device)
            output = model(image)
            probabilities = nn.functional.softmax(output[0], dim=0)
            confidence, predicted_class = torch.max(probabilities, 0)
            return class_names[predicted_class.item()], confidence.item()



