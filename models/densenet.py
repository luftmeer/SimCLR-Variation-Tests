"""Helper function return supported DenseNet.

    Raises:
        KeyError: Requested model is not supported.

    Returns:
        torchvision.models.resnet.ResNet: Requested DenseNet Model available from the torchvision library.
"""
from torchvision import models
DENSENET_ENCODERS = ['densenet121']

def get_densenet(name: str) -> models.densenet.DenseNet:
    densenets = {
        "densenet121": models.densenet121(),
    }
    
    if name not in densenets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    
    encoder = densenets[name]
    n_features = encoder.classifier.in_features
    
    return encoder, n_features