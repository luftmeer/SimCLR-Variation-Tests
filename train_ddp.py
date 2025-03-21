import argparse

from simclr.simclr import SimCLR
from simclr.transform import SimCLRTransform
from simclr.loss import NTXentLoss
import torch
from models.encoder import get_encoder
from models import resnet, densenet
import tqdm
from utils.dataset_loader import get_dataset
from utils import loader, csv_metrics
import yaml

# DDP
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from torch.distributed.elastic.multiprocessing.errors import record


def ddp_setup():
   torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
   init_process_group(backend="nccl")

def train(model, optimizer, loss_fn, train_loader, local_rank):
    for i, (augmentations, _) in tqdm.tqdm(enumerate(train_loader), desc="Training", total=len(train_loader)):
        optimizer.zero_grad()
        
        _, zs = model(augmentations)
        
        loss = loss_fn(zs)
        loss.backward()
        optimizer.step()
        
        loss_epoch += loss.item()
        
        if i % 50 == 0:
            print(f"Step [{i}/{len(train_loader)}]\t Loss: {loss.item()}")



@record
def main(args):
    ddp_setup()
    
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])

    # Randomness
    torch.manual_seed(args.seed)
    if local_rank == 0 and global_rank == 0:
        csv_metric = csv_metrics.CSV_Metric(args)
    
    
    train_dataset = get_dataset(dataset_name=args.dataset_name, train=args.dataset_train, image_size=args.resize, augmentations=args.augmentations, HF_TOKEN=args.HF_TOKEN)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            sampler=train_sampler,
        )

    encoder, n_features = get_encoder(args.encoder)

    if args.resume:
        start_epoch = args.start_epoch
    else:
        model = SimCLR(encoder=encoder, n_features=n_features, projection_dim=args.projection_dim, image_size=args.resize, batch_size=args.batch_size, device=local_rank).to(local_rank)
        start_epoch = 0
    
    loss_fn = NTXentLoss(batch_size=args.batch_size, device=local_rank).to(local_rank)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank])
    model.to(local_rank)
    
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        if local_rank == 0 and global_rank == 0:
            csv_metric.start()
        loss_epoch = 0
        
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        loss_epoch = train(model, optimizer, loss_fn, train_loader, local_rank)
        
        if local_rank == 0 and global_rank == 0:        
            csv_metric.end()
        
        print(f'Epoch {epoch+1} | Global Rank {global_rank} | Loss: {loss_epoch}')
        
        if local_rank == 0 and global_rank == 0:
            csv_metric.write(epoch=epoch, loss=loss_epoch)
            
        if (epoch+1) % args.save_every_epoch == 0 and global_rank == 0:
            print(f"Saving model at Epoch {epoch+1}")
            loader.save_model(model=model, optimizer=optimizer, loss=loss_fn, dataset_name=args.dataset_name, epoch=epoch, encoder=args.encoder, args=args, csv_metric=csv_metric)
    
    print(f"Saving final model at Epoch {epoch+1}")
    loader.save_model(model=model, optimizer=optimizer, loss=loss_fn, dataset_name=args.dataset_name, epoch=epoch, encoder=args.encoder, args=args, csv_metric=csv_metric)
    destroy_process_group()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Adjusted SimCLR implementation to analyse input variations",
    )
    
    parser.add_argument('--config', '-c', type=str, default='./config/default.yaml', help='Config file to run the training')
    
    parser.add_argument('--checkpoint', type=object, default=None, help='Add your checkpoint here if you want to resume a previous training.')
    
    # Parse arguments known up till here, the rest via config file
    args = parser.parse_known_args()[0]
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        for elem in config:
            k, v = elem.popitem()
            parser.add_argument(f"--{k}", default=v, type=type(v))    

    args = parser.parse_args()
    
    main(args)