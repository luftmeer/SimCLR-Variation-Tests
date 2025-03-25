import argparse
import os
from filelock import FileLock
import csv
import torch.distributed as dist
import socket

def log_loss(epoch: int, loss: object, args: argparse.Namespace, elapsed_time: float):
    log_file = f"{'_'.join(str(elem) for elem in [args.encoder, args.optimizer, args.epochs, args.batch_size, args.augmentations, args.projection_dim, args.temperature])}.csv"

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    
        row = {
            'epoch': epoch+1,
            'rank': rank,
            'global_rank': int(os.environ["RANK"]),
            'world_size': world_size,
            'loss': float(loss),
            'host': socket.gethostname(),
            'elapsed_time': elapsed_time,
        }

    else:
        row = {
            'epoch': epoch+1,
            'loss': float(loss),
            'host': socket.gethostname(),
            'elapsed_time': elapsed_time,
        }

    # Shared file path (can be an absolute path if needed)
    if args.slurm_job_id:
        log_path = os.path.join(os.getcwd(), 'metrics',  args.dataset_name, str(args.slurm_job_id), log_file)
    else:
        log_path = os.path.join(os.getcwd(), 'metrics', args.dataset_name, log_file)
    lock_path = log_path + ".lock"

    # Use FileLock to prevent simultaneous write
    with FileLock(lock_path):
        file_exists = os.path.isfile(log_path)
        with open(log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                f.write(f"#{str(args)}\n")
                writer.writeheader()
            writer.writerow(row)


def log_evaluation(epoch: int, loss, accuracy, args: argparse.Namespace, elapsed_time: float, cpt_epoch: int):
    log_file = f"{'_'.join(str(elem) for elem in [args.encoder, args.optimizer, args.epochs, args.batch_size, args.augmentations, args.projection_dim, args.temperature, f"cpt_epoch{cpt_epoch}"])}.csv"
    
    row = {
            'epoch': epoch+1,
            'loss': float(loss),
            'accuracy': accuracy,
            'host': socket.gethostname(),
            'elapsed_time': elapsed_time,
        }
    
    if args.slurm_job_id:
        log_path = os.path.join(os.getcwd(), 'metrics',  args.dataset_name, 'linear_evaluation', str(args.slurm_job_id), log_file)
    else:
        log_path = os.path.join(os.getcwd(), 'metrics', args.dataset_name, 'linear_evaluation', log_file)


    file_exists = os.path.isfile(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            f.write(f"#{str(args)}\n")
            writer.writeheader()
        writer.writerow(row)