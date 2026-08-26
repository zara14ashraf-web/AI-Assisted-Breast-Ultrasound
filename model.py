import torch
import torch.nn as nn
import timm


class SharedDualEfficientNetB3(nn.Module):
    """
    Dual-view EfficientNet-B3 model used for BUS-BRA classification.

    Inputs:
        full_img:  Full ultrasound image, shape [B, 3, H, W]
        crop_img:  Lesion-focused crop, shape [B, 3, H, W]

    Output:
        Logits for:
            0 = Benign
            1 = Malignant
    """

    def __init__(self):
        super().__init__()

        # Full-image branch
        self.full_branch = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=0
        )

        # Lesion-crop branch
        self.crop_branch = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=0
        )

        # EfficientNet-B3 feature dimension
        feature_dim = 1536

        # Fusion + classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 2)
        )

    def forward(self, full_img, crop_img):

        full_features = self.full_branch(full_img)
        crop_features = self.crop_branch(crop_img)

        combined = torch.cat(
            [full_features, crop_features],
            dim=1
        )

        return self.classifier(combined)


def load_model(checkpoint_path, device=None):
    """
    Load the trained BUS-BRA Dual-View EfficientNet-B3 model.
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = SharedDualEfficientNetB3().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # Handle both raw state_dict and wrapped checkpoints
    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:
        raise ValueError(
            "Unexpected checkpoint format."
        )

    # Strict loading is intentional:
    # deployment must use exactly the trained architecture.
    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    return model
