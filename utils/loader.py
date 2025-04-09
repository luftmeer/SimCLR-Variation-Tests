import torch
import os
from datetime import datetime
import argparse

# Base Folder for Checkpoints
CHECKPOINTS_FOLDER = 'checkpoints'
EVAL_CPT_FOLDER = 'linear_eval_logs_'

def load_model(path:str, device, eval:bool=False, args_only:bool=False) -> torch.nn:
    if not os.path.exists(path):
        raise FileNotFoundError("File or Path not correct")

    if isinstance(device, torch.device):
        cpt = torch.load(path, map_location=device.type, weights_only=False)
    elif isinstance(device, str):
        cpt = torch.load(path, map_location=device, weights_only=False)
    
    if eval:
        return cpt['model_state_dict'], cpt['epoch']
    
    if args_only:
        return cpt['args']
    
    return cpt


def save_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, dataset_name: str, epoch: int, encoder: str, args:argparse.Namespace, base_folder: str=None) -> None:
    path = os.path.join('./', 
                        base_folder if base_folder else '',
                        str(args.slurm_job_id) if args.slurm_job_id else '',
                        CHECKPOINTS_FOLDER)
    os.makedirs(path, exist_ok=True)
    
    filename_content = [args.slurm_job_id,
                        args.dataset_name,
                        args.encoder, 
                        args.optimizer, 
                        args.batch_size, 
                        args.augmentations, 
                        args.projection_dim, 
                        args.temperature,
                        epoch+1
                    ]
    
    filename = f"{path}" + f"/{datetime.now().strftime('%Y%m%d%H%M%S')}_{'_'.join(str(elem) for elem in filename_content)}.cpt"
    
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

def save_evaluation(simclr_model: torch.nn.Module, model: torch.nn.Module, args: argparse.Namespace, cpt_epoch: int, epoch: int, base_folder: str=None) -> None:
    path = os.path.join('./', 
                        base_folder if base_folder else '',
                        str(args.slurm_job_id) if args.slurm_job_id else '',
                        f"{EVAL_CPT_FOLDER}{args.optimizer}",
                        f'cpt_epoch{cpt_epoch}')
    os.makedirs(path, exist_ok=True)
     
    filename_content = [args.encoder,
                        args.optimizer,
                        f'cpt_epoch{cpt_epoch}',
                        f'epoch{epoch+1}',
                    ]
    
    filename = path + f"/{args.slurm_job_id}_{'_'.join(str(elem) for elem in filename_content)}.eval.cpt"
    
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

def save_model_eval(simclr_model: torch.nn.Module, model: torch.nn.Module, args: argparse.Namespace, cpt_epoch: int, epoch: int, optimizer: torch.optim.Optimizer, best_model: bool=False, base_folder: str=None) -> None:
    path = os.path.join('./', 
                        base_folder if base_folder else '',
                        str(args.slurm_job_id) if args.slurm_job_id else '',
                        f"{EVAL_CPT_FOLDER}{args.optimizer}",
                        f'cpt_epoch{cpt_epoch}')
    os.makedirs(path, exist_ok=True)
    
    filename_content = ['linear-evaluation',
                        args.encoder,
                        args.optimizer,
                        f'cpt_epoch{cpt_epoch}',
                        '' if best_model else f'epoch{epoch+1}',
                    ]
    if not best_model:
        filename = path + f"/{'_'.join(str(elem) for elem in filename_content)}.cpt"
    else:
        filename = path + f"/{'_'.join(str(elem) for elem in filename_content)}.cpt.best"
    
    
    torch.save(
            {
                "epoch": epoch,
                "simclr_model_state_dict": simclr_model.state_dict(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": args,
                "dataset_name": args.dataset_name,
                "cpt_epoch": cpt_epoch,
            },
            filename,
        )
    
    return