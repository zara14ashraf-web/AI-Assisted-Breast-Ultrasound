import torch
import torch.nn as nn
import timm


class EfficientNetB3Classifier(nn.Module):
    """
    Final BUS-BRA EfficientNet-B3 classifier.

    Single-image input:
        Original breast ultrasound image

    Classes:
        0 = Benign
        1 = Malignant
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=0,
        )

        self.classifier = nn.Sequential(
            nn.Identity(),
            nn.Linear(
                1536,
                num_classes,
            ),
        )

    def forward(self, x):

        features = self.backbone(x)

        logits = self.classifier(
            features
        )

        return logits
