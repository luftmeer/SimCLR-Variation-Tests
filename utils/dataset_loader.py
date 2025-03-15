import torchvision
from datasets import load_dataset
import huggingface_hub
from simclr.transform import SimCLRTransform
import os

DATASETS = ['CIFAR10', 'STL10', 'Imagenette', 'tiny-imagenet']

def get_dataset(dataset_name: str='CIFAR10', train: bool=True, image_size: int=224, augmentations: int=4, **kwargs):
    transform = SimCLRTransform(size=image_size, n=augmentations)
    root_dir = './data'
    assert dataset_name in DATASETS
    match dataset_name:
        case 'CIFAR10':
            ds = torchvision.datasets.CIFAR10(
                root=root_dir,
                train=train,
                transform=transform,
                download=True,
            )
            
        case 'STL10':
            ds = torchvision.datasets.STL10(
                root=root_dir,
                split='unlabeled',
                transform=transform,
                download=True,
            )
            
        case 'Imagenette':
            ds = torchvision.datasets.Imagenette(
                root=root_dir,
                split='train',
                size='full',
                transform=transform,
                download=True,
            )
            
        case 'tiny-imagenet':
            raise NotImplementedError("TODO!")
            if 'HF_TOKEN' in os.environ:
                token = os.environ.get('HF_TOKEN')
            elif 'HF_TOKEN' in kwargs.keys():
                token = kwargs['HF_TOKEN']
            else:
                raise AttributeError(f"Token for Hugging Face not set or given.")
            
            user = huggingface_hub.login(token)
            ds = load_dataset("zh-plus/tiny-imagenet")
            ds.save_to_disk(root_dir)
            
    return ds
            
