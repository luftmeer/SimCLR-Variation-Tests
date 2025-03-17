"""Helper function return supported DenseNet.

    Raises:
        KeyError: Requested model is not supported.

    Returns:
        torchvision.models.resnet.ResNet: Requested DenseNet Model available from the torchvision library.
"""
from torchvision import models
from torch import nn
DENSENET_ENCODERS = ['densenet121', 'densenet161', 'densenet169', 'densenet201']

def get_densenet(name: str, widening: int) -> models.densenet.DenseNet:
    densenets = {
        'densenet121': models.densenet121(),
        'densenet161': models.densenet161(),
        'densenet169': models.densenet169(),
        'densenet201': models.densenet201(),
    }
    
    if name not in densenets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    
    encoder = densenets[name]
    
    if widening > 1:
        encoder = modify_densenet_layers(encoder, widening)

    n_features = encoder.classifier.in_features
    
    return encoder, n_features


def modify_densenet_layers(encoder: nn.Module, width_multiplier=2):
    """
    Modify a torchvision DenseNet model to widen all layers based on the given width_multiplier.
    """
    # Track the number of features as we go deeper
    original_growth_rate = int(encoder.features.conv0.out_channels / 2)
    print(original_growth_rate)
    num_features = encoder.features.conv0.out_channels * width_multiplier  # Start with the widened conv0 output

    # Get the original DenseNet growth rate (default: 32)
    new_growth_rate = original_growth_rate * width_multiplier

    encoder.features.conv0 = nn.Conv2d(
        in_channels=3,
        out_channels=num_features,  # Increase initial channels
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    encoder.features.norm0 = nn.BatchNorm2d(num_features)  # Match widened conv0

    

    for module in encoder.features.children():
        if isinstance(module, models.densenet._DenseBlock):
            for layer in module.children():
                if isinstance(layer, models.densenet._DenseLayer):

                    layer.norm1 = nn.BatchNorm2d(num_features)  # 🔥 Fix for mismatch
                    
                    layer.conv1 = nn.Conv2d(
                        in_channels=num_features,
                        out_channels=layer.conv1.out_channels * width_multiplier,
                        kernel_size=1,
                        bias=False,
                    )
                    
                    layer.norm2 = nn.BatchNorm2d(layer.conv1.out_channels)  # Ensure BatchNorm matches
                    layer.conv2 = nn.Conv2d(
                        in_channels=layer.conv1.out_channels,
                        out_channels=layer.conv2.out_channels * width_multiplier,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    )
                    
                    num_features += new_growth_rate  # FIX: Ensure we track correct feature size dynamically
                    
        if isinstance(module, models.densenet._Transition):
            module.norm = nn.BatchNorm2d(num_features)  # FIX: Ensure BatchNorm matches
            module.conv = nn.Conv2d(
                in_channels=num_features,
                out_channels=module.conv.out_channels * width_multiplier,
                kernel_size=1,
                bias=False
            )
            num_features = module.conv.out_channels

    
    encoder.features.norm5 = nn.BatchNorm2d(encoder.classifier.in_features * width_multiplier)  # Match widened conv0

    
    in_features = encoder.classifier.in_features * width_multiplier  # FIXED
    out_features = encoder.classifier.out_features
    encoder.classifier = nn.Linear(in_features, out_features)

    return encoder