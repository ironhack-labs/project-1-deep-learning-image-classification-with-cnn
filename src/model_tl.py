import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def build_resnet18(num_classes: int = 10, freeze_backbone: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

def unfreeze_layer4(model):
    for name, p in model.named_parameters():
        if name.startswith("layer4") or name.startswith("fc"):
            p.requires_grad = True