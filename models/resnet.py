"""Helper function return supported ResNets.

    Raises:
        KeyError: Requested model is not supported.

    Returns:
        torchvision.models.resnet.ResNet: Requested ResNet Model available from the torchvision library.
"""
from torchvision import models
from torch import nn
RESNET_ENCODERS = ['resnet18', 'resnet50']

def get_resnet(name: str, widening: int=1) -> models.resnet.ResNet:
    resnets = {
        "resnet18": models.resnet18(),
        "resnet50": models.resnet50(),
    }
    
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    
    encoder = resnets[name]
    
    if widening > 1:
        encoder = modify_bottleneck_blocks(encoder, widening)
    
    n_features = encoder.fc.in_features
    
    return encoder, n_features


def modify_bottleneck_blocks(encoder: nn.Module, width_multiplier: int):
    """Modify the bottleneck blocks to increase width"""
    encoder.conv1 = nn.Conv2d(3, 64 * width_multiplier, kernel_size=7, stride=2, padding=3, bias=False)
    encoder.bn1 = nn.BatchNorm2d(64 * width_multiplier)  # Fix BatchNorm issue

    for name, module in encoder.named_modules():
            if isinstance(module, models.resnet.Bottleneck):
                in_channels = module.conv1.in_channels * width_multiplier
                out_channels = module.conv1.out_channels * width_multiplier

                # Scale up all conv layers in the Bottleneck block
                module.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
                module.bn1 = nn.BatchNorm2d(out_channels)  # Fix BatchNorm issue

                module.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=module.conv2.stride, padding=1, bias=False)
                module.bn2 = nn.BatchNorm2d(out_channels)  # Fix BatchNorm issue

                module.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
                module.bn3 = nn.BatchNorm2d(out_channels * 4)  # Fix BatchNorm issue

                # Modify the shortcut (downsampling layer) to match new width
                if module.downsample is not None:
                    module.downsample[0] = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=module.downsample[0].stride, bias=False)
                    module.downsample[1] = nn.BatchNorm2d(out_channels * 4)  # Fix BatchNorm issue
    
    in_features = encoder.fc.in_features
    num_classes = encoder.fc.out_features
    encoder.fc = nn.Linear(in_features * width_multiplier, num_classes)
    return encoder