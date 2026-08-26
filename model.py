import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3


class EfficientNetB3Classifier(nn.Module):
    """
    BUS-BRA EfficientNet-B3 classifier.

    Classes:
        0 = Benign
        1 = Malignant
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = efficientnet_b3(
            weights=None
        )

        # Original checkpoint uses:
        # classifier.1.weight -> (2, 1536)
        # classifier.1.bias   -> (2,)
        self.backbone.classifier[1] = nn.Linear(
            1536,
            num_classes
        )

    def forward(self, x):
        return self.backbone(x)

    def get_cam_target_layer(self):
        """
        Target layer for Grad-CAM.
        """
        return self.backbone.features[-1]
