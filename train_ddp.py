import argparse

from simclr.simclr import SimCLR
from simclr.loss import NTXentLoss
import torch
from models.encoder import get_encoder
import tqdm
from utils.dataset_loader import get_dataset
from utils import loader
from utils.log_loss import log_loss
import yaml
import time
from itertools import combinations
import datetime

# DDP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import os
from torch.distributed.elastic.multiprocessing.errors import record

# LARS Optimizer
from flash.core.optimizers import LARS

# Logging and Monitoring
from utils.logger import TrainingMonitor

# Base folder for all runs
BASE_FOLDER = 'runs'

def ddp_setup():
   torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
   init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=20))

def gather_from_world(tensor: torch.Tensor) -> torch.Tensor:
    """Gather all projections from the other nodes and GPUs.

    Args:
        tensor (torch.Tensor): Projection of the current GPU

    Returns:
        torch.Tensor: Projections of the current GPU plus the projections of the other GPUS attached
    """
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    
    # Replace current rank’s tensor in gathered list with the original one
    # so that autograd can track it
    gathered[dist.get_rank()] = tensor

    return torch.cat(gathered, dim=0)

def train(model, optimizer, loss_fn, train_loader, local_rank, global_rank, monitor, epoch, args):
    all_embeddings = [[] for _ in range(args.augmentations)]
    all_projections = [[] for _ in range(args.augmentations)]
    all_labels = []
    total_loss = 0
    optimizer.zero_grad() # In case Gradient Accumulation is enabled and won't execute in first step
    for i, (augmentations, labels) in tqdm.tqdm(enumerate(train_loader), desc="Training", total=len(train_loader)):
        labels = labels.to(local_rank)
        all_labels.append(gather_from_world(labels).detach().cpu())
        if args.ga and (i+1) % args.ga_count == 0 or not args.ga or (i+1) == len(train_loader):
            optimizer.zero_grad()
        
        hs, zs = model(augmentations)

        
        # For logging purposes only
        hs_all = []
        for i, h in enumerate(hs):
            full_emb = gather_from_world(h)
            hs_all.append(full_emb)
            all_embeddings[i].append(full_emb.detach().cpu())
        
        zs_all = []
        for i, z in enumerate(zs):
            full_proj = gather_from_world(z)
            zs_all.append(full_proj)
            all_projections[i].append(full_proj.detach().cpu())
        
        for comb_nr, (z_i, z_j) in enumerate(combinations(zs_all, 2)):
            loss, logits, sim, positives, negatives = loss_fn(z_i, z_j)
            total_loss += loss.item()
            monitor.log_logits(i, epoch, comb_nr, logits)
            monitor.log_pos_neg_samples(positives, negatives, comb_nr)
            monitor.log_losses(loss.item(), comb_nr)
            if i % 50 == 0:
                print(f"Step [{i}/{len(train_loader)}]\t Loss: {loss.item()} | Combination {comb_nr}")
        
        loss.backward()

        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        monitor.log_gradient(model, optimizer, i, epoch)
        
        # Gradient Accumulation
        # First Case: Gradient Accumulation is active and the n-th batch is rached which is divisible by ga_count
        # Second Case: Gradient Accumulation is not available -> always do the step
        # Third Case: The current batch is the last one -> always optimize
        if args.ga and (i+1) % args.ga_count == 0 or not args.ga or (i+1) == len(train_loader):
            optimizer.step()
        

    return total_loss / len(train_loader), sim, all_embeddings, all_projections, all_labels


@record
def main(args):
    ddp_setup()
    
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    device = f'cuda:{local_rank}'
    
    cpt = None
    # If resuming -> overwrite config with "old" one. Keep checkpoint and resume value
    if args.resume:
        cpt_path = args.checkpoint
        
        cpt = loader.load_model(path=args.checkpoint, device=device)
        args = cpt['args']
        args.checkpoint = cpt_path
        args.resume = True # Reset -> in previous it was False

    # Randomness
    torch.manual_seed(args.seed)
    
    # Monitoring
    monitor = TrainingMonitor(
        save_dir=f'{BASE_FOLDER}/{args.slurm_job_id}/',
        plot_every=100,
        batch_size=args.batch_size * dist.get_world_size(),
        maxlen=500,
        enabled=True,
        rank=global_rank,
        n_augments=args.augmentations,
    )
    
    
    train_dataset = get_dataset(dataset_name=args.dataset_name, train=args.dataset_train, image_size=args.resize, augmentations=args.augmentations, HF_TOKEN=args.HF_TOKEN, args=args)

    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=local_rank, shuffle=True)

    tiny_subset = torch.utils.data.Subset(train_dataset, indices=list(range(16)))
    
    train_loader = torch.utils.data.DataLoader(
            tiny_subset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            sampler=train_sampler,
        )

    encoder, n_features = get_encoder(encoder=args.encoder, widening=args.widening)

    model = SimCLR(encoder=encoder, n_features=n_features, projection_dim=args.projection_dim, image_size=args.resize, batch_size=args.batch_size, device=local_rank).to(local_rank)
    
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None
    elif args.optimizer == 'LARS':
        batch_size = args.batch_size
        if dist.is_initialized():
            batch_size *= dist.get_world_size()
        
        if args.ga:
            batch_size *= args.ga_count
            
        optimizer = LARS(model.parameters(), lr=0.3*(batch_size/256), weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)
    
    if args.resume:
        start_epoch = cpt['epoch']
        model.load_state_dict(cpt['model_state_dict'])
        optimizer.load_state_dict(cpt['optimizer'])
        
    else:
        start_epoch = 0
    
    loss_fn = NTXentLoss(batch_size=args.batch_size*dist.get_world_size(), device=local_rank).to(local_rank)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank])
    model.to(local_rank)
    

    if args.debug:
        # Anomaly detection -> In case of NaN resulting from the loss function
        torch.autograd.set_detect_anomaly(True)

    model.train()    
    for epoch in range(start_epoch, args.epochs):        
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        start = time.time()
        
        loss_epoch, sim, all_embeddings, all_projections, all_labels = train(model, optimizer, loss_fn, train_loader, local_rank, global_rank, monitor, epoch, args)
        
        end = time.time()
        
        print(f'Epoch {epoch+1} | Global Rank {global_rank} | Local Rank {local_rank} | Loss: {loss_epoch}')
        
        monitor.log_epoch(epoch, optimizer.param_groups[0]["lr"], end-start, sim)
        monitor.log_tsne_embeddings(all_embeddings, all_projections, all_labels, epoch)
        monitor.check_embedding_collapse(all_embeddings, epoch)
        
        if scheduler:
            scheduler.step()
        
        if args.metrics:
            log_loss(epoch=epoch, loss=loss_epoch, args=args, elapsed_time=end-start, base_folder=BASE_FOLDER)
            
        if (epoch+1) % args.save_every_epoch == 0 and global_rank == 0:
            print(f"Saving model at Epoch {epoch+1}")
            loader.save_model(model=model, optimizer=optimizer, dataset_name=args.dataset_name, epoch=epoch, encoder=args.encoder, args=args, base_folder=BASE_FOLDER)
    
    if global_rank == 0:
        print(f"Saving final model at Epoch {epoch+1}")
        loader.save_model(model=model, optimizer=optimizer, loss=loss_fn, dataset_name=args.dataset_name, epoch=epoch, encoder=args.encoder, args=args)
    destroy_process_group()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Adjusted SimCLR implementation to analyse input variations",
    )
    
    parser.add_argument('--config', '-c', type=str, default='./config/default.yaml', help='Config file to run the training')
    
    parser.add_argument('--checkpoint', type=str, default=None, help='Add your checkpoint here if you want to resume a previous training.')
    
    parser.add_argument('--slurm_job_id', type=int, default=None)
    
    parser.add_argument('--debug', action='store_true')
    
    parser.add_argument('--resume', action='store_true')
    
    parser.add_argument('--ga', action='store_true')
    
    # Parse arguments known up till here, the rest via config file
    args = parser.parse_known_args()[0]
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        for elem in config:
            k, v = elem.popitem()
            if k in ['lr', 'weight_decay', 'eps']:
                parser.add_argument(f'--{k}', default=v, type=float)
            elif k == 'ga':
                continue
            else:
                parser.add_argument(f"--{k}", default=v, type=type(v))    

    args = parser.parse_args()
    
    main(args)