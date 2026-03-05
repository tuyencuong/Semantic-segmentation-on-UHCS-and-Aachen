"""Model registry for unified Aachen/UHCS segmentation training and inference."""

from .unet import UNetVgg16
from .unet_uhcs_legacy import UNetVgg16UHCSLegacy


MODEL_VARIANTS = {
    "aachen_baseline": UNetVgg16,
    "uhcs_legacy": UNetVgg16UHCSLegacy,
}

# Backward aliases used by old scripts.
MODEL_VARIANTS["unet"] = MODEL_VARIANTS["aachen_baseline"]


def infer_model_variant_from_state_dict(state_dict):
    """Infer the model variant from checkpoint keys."""
    keys = list(state_dict.keys())

    if any("module." in k for k in keys):
        keys = [k[7:] if k.startswith("module.") else k for k in keys]

    # UHCS legacy markers
    uhcs_markers = (
        "bottleneck_attention",
        "align_x4",
        "align_x3",
        "align_x2",
        "align_x1",
        "inc.1.fc1.weight",
        "up1.attention",
    )
    if any(any(m in k for m in uhcs_markers) for k in keys):
        return "uhcs_legacy"

    # Aachen baseline markers
    aachen_markers = ("aspp", "eca_bottleneck")
    if any(any(m in k for m in aachen_markers) for k in keys):
        return "aachen_baseline"

    # Safe default to baseline architecture.
    return "aachen_baseline"


def build_model(n_classes, model_variant="aachen_baseline", encoder_pretrained=True):
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(
            f"Unknown model_variant='{model_variant}'. "
            f"Available: {sorted(set(MODEL_VARIANTS.keys()))}"
        )
    model_cls = MODEL_VARIANTS[model_variant]
    return model_cls(n_classes=n_classes, encoder_pretrained=encoder_pretrained)

