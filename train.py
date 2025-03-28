import argparse

from simclr.simclr import SimCLR
from simclr.transform import SimCLRTransform
from simclr.loss import NTXentLoss
import torch
from models.encoder import get_encoder
from models import resnet, densenet
import tqdm
from utils.dataset_loader import get_dataset
from utils import loader
from utils.log_loss import log_loss
import yaml
from filelock import FileLock
import socket
import csv
import time
from torch.amp import autocast, GradScaler
import os
# LARS Optimizer
from flash.core.optimizers import LARS

def train(model, optimizer, loss_fn, train_loader, scaler, args):
    total_loss = 0
    for i, (augmentations, _) in tqdm.tqdm(enumerate(train_loader), desc="Training", total=len(train_loader)):
        optimizer.zero_grad()
        
        
        with autocast(device_type='cuda'):
            _, zs = model(augmentations)
            zs = [z.float() for z in zs]
            
            loss = loss_fn(zs)
        if torch.is_autocast_enabled():
            scaler.scale(loss).backward()
            if args.ga and i % args.ga_count == 0 or not args.ga or i+1 == len(train_loader):
                scaler.step(optimizer)
            scaler.update()
        else:
            
            loss.backward()
        
        # Gradient Accumulation
        # First Case: Gradient Accumulation is active and the n-th batch is rached which is divisible by ga_count
        # Second Case: Gradient Accumulation is not available -> always do the step
        # Third Case: The current batch is the last one -> always optimize
            if args.ga and i % args.ga_count == 0 or not args.ga or i+1 == len(train_loader):
                optimizer.step()
        
        
        total_loss += loss.item()
        
        if i % 50 == 0:
            print(f"Step [{i}/{len(train_loader)}]\t Loss: {loss.item()}")

    return total_loss

def main(args):
    # Randomness
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    train_dataset = get_dataset(dataset_name=args.dataset_name, train=args.dataset_train, image_size=args.resize, augmentations=args.augmentations, HF_TOKEN=args.HF_TOKEN, args=args)

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    encoder, n_features = get_encoder(encoder=args.encoder, widening=args.widening)
    model = SimCLR(encoder=encoder, n_features=n_features, projection_dim=args.projection_dim, image_size=args.resize, batch_size=args.batch_size, device=device).to(device)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None
    elif args.optimizer == 'LARS':
        optimizer = LARS(model.parameters(), lr=0.3*(args.batch_size/256), weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)
    
    if args.resume:
        cpt = loader.load_model(path=args.checkpoint, device=device, eval=False)
        start_epoch = cpt['epoch']
        model.load_state_dict(cpt['model_state_dict'])
        optimizer.load_state_dict(cpt['optimizer'])
            
    else:
        start_epoch = 0
    
    loss_fn = NTXentLoss(batch_size=args.batch_size, device=device).to(device)

    model.to(device)
    
        
    if args.half_precision:
        scaler = GradScaler()
    else:
            scaler = None

    if args.debug:
        # Anomaly detection -> In case of NaN resulting from the loss function
        torch.autograd.set_detect_anomaly(True)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        
        start = time.time()
        
        loss_epoch = train(model, optimizer, loss_fn, train_loader, scaler, args)
        
        end = time.time()
        
        print(f'Epoch {epoch+1} | Loss: {loss_epoch}')
        
        if scheduler:
            scheduler.step()
        
        if args.metrics:
            log_loss(epoch=epoch, loss=loss_epoch, args=args, elapsed_time=end-start)
            
        if (epoch+1) % args.save_every_epoch == 0:
            print(f"Saving model at Epoch {epoch+1}")
            loader.save_model(model=model, optimizer=optimizer, loss=loss_fn, dataset_name=args.dataset_name, epoch=epoch, encoder=args.encoder, args=args)
    
    print(f"Saving final model at Epoch {epoch+1}")
    loader.save_model(model=model, optimizer=optimizer, loss=loss_fn, dataset_name=args.dataset_name, epoch=epoch, encoder=args.encoder, args=args)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Adjusted SimCLR implementation to analyse input variations",
    )
    
    parser.add_argument('--config', '-c', type=str, default='./config/default.yaml', help='Config file to run the training')
    
    parser.add_argument('--checkpoint', type=str, default=None, help='Add your checkpoint here if you want to resume a previous training.')
    
    parser.add_argument('--slurm_job_id', type=int, default=None)
    
    parser.add_argument('--debug', action='store_true')
    
    # Parse arguments known up till here, the rest via config file
    args = parser.parse_known_args()[0]
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        for elem in config:
            k, v = elem.popitem()
            parser.add_argument(f"--{k}", default=v, type=type(v))    

    args = parser.parse_args()
    
    main(args)