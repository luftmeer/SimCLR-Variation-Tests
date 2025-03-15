from torchvision import transforms
import torch.nn as nn
import torch

class SimCLRTransform:
    def __init__(self, size: int, s: float=1.0, n: int=2, eval: bool=False):
        self.n = n
        self.size = size
        self.eval = eval
        
        # Pre-defined augmentations
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=self.size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor()
            ]
        )
        # Evaluation Transform. 
        self.eval_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=self.size)
            ]
        )
    
    def __call__(self, img):
        augmentations = []
        if not self.eval:
            for _ in range(self.n):
                augmentations.append(self.transform(img))
        else:
            augmentations = [self.eval_transform(img)]
        
        return augmentations # returns n transformed images in one list. The DataLoader will split it up and create n batch sets.