import torch
import os
from datetime import datetime
import argparse

# Base Folder for Checkpoints
CHECKPOINTS_FOLDER = './checkpoints'

def load_model(path:str, device:torch.device, eval:bool=False) -> torch.nn:
    if not os.path.exists(path):
        raise FileNotFoundError("File or Path not correct")

    cpt = torch.load(path, map_location=device.type, weights_only=False)
    
    if eval:
        return cpt['model_state_dict'], cpt['epoch']
    
    return cpt


def save_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss: object, dataset_name: str, epoch: int, encoder: str, args:argparse.Namespace) -> None:
    if not os.path.exists(CHECKPOINTS_FOLDER):
        os.makedirs(CHECKPOINTS_FOLDER)

    # Create sub folder for dataset name in checkpoint folder, if it doesn't exist yet
    if not os.path.exists(f"{CHECKPOINTS_FOLDER}/{dataset_name}/"):
        os.makedirs(f"{CHECKPOINTS_FOLDER}/{dataset_name}/")
    
    # Subfolder for Slurm Job
    if args.slurm_job_id:
        if not os.path.exists(f"{CHECKPOINTS_FOLDER}/{dataset_name}/{args.slurm_job_id}/"):
            os.makedirs(f"{CHECKPOINTS_FOLDER}/{dataset_name}/{args.slurm_job_id}/")
    
    filename_content = [args.encoder, 
                        args.optimizer, 
                        args.batch_size, 
                        args.augmentations, 
                        args.projection_dim, 
                        args.temperature,
                        epoch+1
                    ]
    
    filename = f"{CHECKPOINTS_FOLDER}/{dataset_name}"
    
    # Subfolder for Slurm Job
    if args.slurm_job_id:
        filename = filename + f"/{args.slurm_job_id}"
    
    filename = filename + f"/{datetime.now().strftime('%Y%m%d%H%M%S')}-{args.slurm_job_id}_{'_'.join(str(elem) for elem in filename_content)}.cpt"
    
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        torch.save(
            {
                "epoch": epoch + 1,
                # +1 since, when starting again, the algorithm should continue with the next epoch and not 'redo' this one
                "model_state_dict": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                #"loss": loss.state_dict(),
                "args": args,
                "dataset_name": dataset_name,
            },
            filename,
        )
    else:
        torch.save(
            {
                "epoch": epoch + 1,
                # +1 since, when starting again, the algorithm should continue with the next epoch and not 'redo' this one
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                #"loss": loss.state_dict(),
                "args": args,
                "dataset_name": dataset_name,
            },
            filename,
        )
    
    return

def save_evaluation(simclr_model: torch.nn.Module, model: torch.nn.Module, args: argparse.Namespace, cpt_epoch: int) -> None:
    if not os.path.exists(CHECKPOINTS_FOLDER):
        os.makedirs(CHECKPOINTS_FOLDER)

    # Create sub folder for dataset name in checkpoint folder, if it doesn't exist yet
    if not os.path.exists(f"{CHECKPOINTS_FOLDER}/{args.dataset_name}/"):
        os.makedirs(f"{CHECKPOINTS_FOLDER}/{args.dataset_name}/")
    
    # Subfolder for Slurm Job
    if args.slurm_job_id:
        if not os.path.exists(f"{CHECKPOINTS_FOLDER}/{args.dataset_name}/{args.slurm_job_id}/"):
            os.makedirs(f"{CHECKPOINTS_FOLDER}/{args.dataset_name}/{args.slurm_job_id}/")
    
    filename_content = [args.encoder,
                        cpt_epoch
                    ]
    
    filename = f"{CHECKPOINTS_FOLDER}/{args.dataset_name}"
    
    # Subfolder for Slurm Job
    if args.slurm_job_id:
        filename = filename + f"/{args.slurm_job_id}"
    
    filename = filename + f"/{datetime.now().strftime('%Y%m%d%H%M%S')}-{args.slurm_job_id}_{'_'.join(str(elem) for elem in filename_content)}.cpt"
    
    torch.save(
            {
                "simclr_model_state_dict": simclr_model.state_dict(),
                "model_state_dict": model.state_dict(),
                "args": args,
                "dataset_name": args.dataset_name,
            },
            filename,
        )
    
    return