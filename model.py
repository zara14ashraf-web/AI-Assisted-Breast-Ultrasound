import torch
import torch.nn as nn
import timm


class SharedDualEfficientNetB3(nn.Module):
    """
    Dual-view EfficientNet-B3 classifier.

    The model processes:
        1. Full ultrasound image
        2. Lesion-focused image

    The two EfficientNet-B3 branches are kept separate to remain
    exactly compatible with the trained Model_Best.pth checkpoint.
    """

    def __init__(self, num_classes=2):
        super().__init__()

        # --------------------------------------------------------
        # FULL-IMAGE BRANCH
        # --------------------------------------------------------

        self.full_branch = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=0,
        )

        # --------------------------------------------------------
        # LESION-FOCUSED BRANCH
        # --------------------------------------------------------

        self.crop_branch = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=0,
        )

        # EfficientNet-B3 feature dimension
        feature_dim = 1536

        # --------------------------------------------------------
        # CLASSIFIER
        # --------------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(
                feature_dim * 2,
                512,
            ),

            nn.BatchNorm1d(512),

            nn.ReLU(
                inplace=True,
            ),

            nn.Dropout(0.30),

            nn.Linear(
                512,
                128,
            ),

            nn.ReLU(
                inplace=True,
            ),

            nn.Dropout(0.20),

            nn.Linear(
                128,
                num_classes,
            ),
        )

    # ============================================================
    # FORWARD
    # ============================================================

    def forward(
        self,
        full_image,
        crop_image,
    ):

        full_features = self.full_branch(
            full_image
        )

        crop_features = self.crop_branch(
            crop_image
        )

        combined_features = torch.cat(
            [
                full_features,
                crop_features,
            ],
            dim=1,
        )

        logits = self.classifier(
            combined_features
        )

        return logits
