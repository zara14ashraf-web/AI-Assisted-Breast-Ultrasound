import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3


class EfficientNetB3Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = efficientnet_b3(weights=None).features

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(1536, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
