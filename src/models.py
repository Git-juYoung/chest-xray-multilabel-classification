import torch.nn as nn
from torchvision import models


def build_resnet50(unfreeze_from, num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    for p in model.parameters():
        p.requires_grad = False

    unfreeze = False
    for name, module in model.named_children():
        if name == unfreeze_from:
            unfreeze = True

        if unfreeze:
            for p in module.parameters():
                p.requires_grad = True

    for p in model.fc.parameters():
        p.requires_grad = True

    return model


def build_efficientnet_b1(unfreeze_last_n_blocks, num_classes):
    model = models.efficientnet_b1(
        weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    for p in model.parameters():
        p.requires_grad = False

    if unfreeze_last_n_blocks > 0:
        for block in model.features[-unfreeze_last_n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

    for p in model.classifier.parameters():
        p.requires_grad = True

    return model