"""Helper function return supported ResNets.

    Raises:
        KeyError: Requested model is not supported.

    Returns:
        torchvision.models.resnet.ResNet: Requested ResNet Model available from the torchvision library.
"""
from torchvision import models
RESNET_ENCODERS = ['resnet18', 'resnet50']

def get_resnet(name: str) -> models.resnet.ResNet:
    resnets = {
        "resnet18": models.resnet18(pretrained=False),
        "resnet50": models.resnet50(pretrained=False),
    }
    
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    
    encoder = resnets[name]
    n_features = encoder.fc.in_features
    
    return encoder, n_features