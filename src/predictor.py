import torch
import torch.nn.functional as F

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