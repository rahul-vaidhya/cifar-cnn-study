"""Two models: a small CNN trained from scratch, and a fine-tuned ResNet-18."""
import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    """3 conv blocks, batch norm, dropout. ~1.2M params."""

    def __init__(self, num_classes=10):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def resnet18_transfer(num_classes=10, freeze_backbone=False):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build(name, **kw):
    if name == "scratch":
        return SmallCNN(**{k: v for k, v in kw.items() if k == "num_classes"})
    if name == "resnet18":
        return resnet18_transfer(**kw)
    raise ValueError(f"unknown model: {name}")
