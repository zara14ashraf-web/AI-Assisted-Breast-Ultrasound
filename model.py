import torch
import torch.nn as nn
import timm


class EfficientNetB3Classifier(nn.Module):
    """
    Latest BUS-BRA Single EfficientNet-B3 classifier.

    Input:
        [B, 3, H, W]

    Output:
        0 = Benign
        1 = Malignant
    """

    def __init__(self):
        super().__init__()

        # IMPORTANT:
        # No "backbone" wrapper.
        # This matches the keys stored in today's Model_Best.pth:
        # features.0.0.weight
        # features.1.0.block...
        self.features = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=2
        )

    def forward(self, x):
        return self.features(x)


def load_model(checkpoint_path, device=None):

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = EfficientNetB3Classifier().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    return model
