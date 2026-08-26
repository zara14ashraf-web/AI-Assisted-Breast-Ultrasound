import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3


class EfficientNetB3Classifier(nn.Module):
    """
    Single EfficientNet-B3 classifier used for BUS-BRA.

    Input:
        Ultrasound image: [B, 3, H, W]

    Output:
        0 = Benign
        1 = Malignant
    """

    def __init__(self):
        super().__init__()

        # Exact EfficientNet-B3 backbone
        backbone = efficientnet_b3(weights=None)

        # Keep the same state-dict structure as the trained checkpoint:
        # features.*
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # EfficientNet-B3 feature dimension = 1536
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1536, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def load_model(checkpoint_path, device=None):

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = EfficientNetB3Classifier()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    # Today's checkpoint is a raw OrderedDict,
    # but this also supports wrapped checkpoints.
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        raise ValueError("Unexpected checkpoint format.")

    # Remove DataParallel prefix if present
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    # EXACT architecture verification
    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model = model.to(device)
    model.eval()

    return model
