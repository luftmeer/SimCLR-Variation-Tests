import torch
import os
from datetime import datetime
import argparse
from .csv_metrics import CSV_Metric

# Base Folder for Checkpoints
CHECKPOINTS_FOLDER = './checkpoints'

def load_model(path:str) -> torch.nn:
    pass


def save_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss: object, dataset_name: str, epoch: int, encoder: str, args:argparse.Namespace, csv_metric: CSV_Metric) -> None:
    if not os.path.exists(CHECKPOINTS_FOLDER):
        os.makedirs(CHECKPOINTS_FOLDER)

    # Create sub folder for dataset name in checkpoint folder, if it doesn't exist yet
    if not os.path.exists(f"{CHECKPOINTS_FOLDER}/{dataset_name}/"):
        os.makedirs(f"{CHECKPOINTS_FOLDER}/{dataset_name}/")
    # datetime_dataset-name_epoch_image-size_augmentations_dataset-size_dataset-distribution.ext 
    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{encoder}_{dataset_name}_{epoch}"
    
    torch.save(
            {
                "epoch": epoch + 1,
                # +1 since, when starting again, the algorithm should continue with the next epoch and not 'redo' this one
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss.state_dict(),
                "args": args,
                "dataset_name": dataset_name,
                "csv_metric": csv_metric,
            },
            filename,
        )
    
    return