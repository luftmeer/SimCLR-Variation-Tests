import argparse

import torchvision
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


def main(args):
    
    # Randomness
    torch.manual_seed(args.seed)
    csv_metric = csv_metrics.CSV_Metric(args)
    if args.resume:
        start_epoch = args.start_epoch
    else:
        start_epoch = 0
        # Init CSV File Metric
        csv_metric = csv_metrics.CSV_Metric(args)
    
    
    
    train_dataset = get_dataset(dataset_name=args.dataset_name, train=args.dataset_train, image_size=args.resize, augmentations=args.augmentations, HF_TOKEN=args.HF_TOKEN)

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            sampler=None,
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, n_features = get_encoder(args.encoder)

    model = SimCLR(encoder=encoder, n_features=n_features, projection_dim=args.projection_dim, image_size=args.resize, batch_size=args.batch_size, device=device).to(device)
    loss_fn = NTXentLoss(batch_size=args.batch_size).to(device)
    
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        csv_metric.start()
        loss_epoch = 0
        for i, (augmentations, _) in tqdm.tqdm(enumerate(train_loader), desc="Training", total=len(train_loader)):
            optimizer.zero_grad()
            
            hs, zs = model(augmentations)
            
            loss = loss_fn(zs)
            loss.backward()
            optimizer.step()
            
            loss_epoch += loss.item()
            
            if i % 50 == 0:
                print(f"Step [{i}/{len(train_loader)}]\t Loss: {loss.item()}")
                
        csv_metric.end()
        print(f'Epoch {epoch+1} | Loss: {loss_epoch}')
        csv_metric.write(epoch=epoch, loss=loss_epoch)
        if epoch % 10 == 0:
            print(f"Saving model at Epoch {epoch+1}")
            loader.save_model(model=model, optimizer=optimizer, loss=loss, dataset_name=args.dataset_name, epoch=epoch, encoder=args.encoder, args=args, csv_metric=csv_metric)
    
    print(f"Saving final model at Epoch {epoch+1}")
    loader.save_model(model=model, optimizer=optimizer, loss=loss, dataset_name=args.dataset_name, epoch=epoch, encoder=args.encoder, args=args, csv_metric=csv_metric)
    
    
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