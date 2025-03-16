import models.densenet as densenet
import models.resnet as resnet

def get_encoder(encoder: str='resnet50', widening: int=1):
    if encoder in densenet.DENSENET_ENCODERS:
        return densenet.get_densenet(encoder, widening)
    elif encoder in resnet.RESNET_ENCODERS:
        return resnet.get_resnet(encoder, widening)
    else:
        raise NotImplemented(f"Requested Encoder not implemented")