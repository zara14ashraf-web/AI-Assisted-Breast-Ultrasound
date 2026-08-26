import torch
import torch.nn as nn
import timm


class EfficientNetB3Classifier(nn.Module):
    """
    Single-view EfficientNet-B3 model for BUS-BRA classification.

    Input:
        image: [B, 3, H, W]

    Output:
        logits:
            0 = Benign
            1 = Malignant
    """

    def __init__(self):
        super().__init__()

        self.model = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=2
        )

    def forward(self, image):
        return self.model(image)


def load_model(checkpoint_path, device=None):

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = EfficientNetB3Classifier().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        raise ValueError(
            "Unexpected checkpoint format."
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.eval()

    return model
