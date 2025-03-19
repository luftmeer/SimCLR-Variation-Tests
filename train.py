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
    
    
    
    train_dataset = get_dataset(dataset_name=args.dataset_name, train=args.train, image_size=args.image_resize, augmentations=args.image_augments, HF_TOKEN=args.HF_TOKEN)

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            sampler=None,
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, n_features = get_encoder(args.encoder)

    model = SimCLR(encoder=encoder, n_features=n_features, projection_dim=args.projection_dim, image_size=args.image_resize, batch_size=args.batch_size, device=device).to(device)
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
    
    # Encoder choice
    parser.add_argument(
        '--encoder', type=str, choices=resnet.RESNET_ENCODERS+densenet.DENSENET_ENCODERS, default='resnet18', help='Learning Encoder'
    )
    
    # Optimizer Choice
    parser.add_argument(
        '--optimizer', type=str, choices=['Adam', 'LARS'], default='Adam', help='Optimizer for the SimCLR Model'
    )
    
    # Batch Size
    parser.add_argument(
        '--batch_size', type=int, default=128
    )
    
    # Image Settings
    parser.add_argument(
        '--image_augments', type=int, default=2, help='The amount of augmentations created and used for the training'
    )
    
    parser.add_argument(
        '--image_resize', type=int, default=224, help='Image width and height for the Resize augmentation'
    )
    
    parser.add_argument(
        '--epochs', type=int, default=100
    )
    
    parser.add_argument(
        '--start_epoch', type=int, default=0
    )
    
    # Projection Dimension for the linear MLP head used to project the embeddings
    parser.add_argument(
        '--projection_dim', type=int, default=128, help='Projection Dimension when calculating the logits'
    )
    
    parser.add_argument(
        '--dataset_name', type=str, choices=['CIFAR10', 'STL10', 'Imagenette', 'tiny-imagenet'], default='CIFAR10', help='Datasat to train on'
    )
    
    # Resume
    parser.add_argument(
        '--resume', action='store_false', help='Resume a already started training. Checkpoint must be given'
    )
    
    parser.add_argument(
        '--checkpoint', type=str, help='Path to a checkpoint'
    )
    
    # Seed
    parser.add_argument(
        '--seed', type=int, default=42
    )
    
    # Activate Train state
    parser.add_argument(
        '--train', action='store_true'
    )
    
    parser.add_argument(
        '--HF_TOKEN', type=str, default=None,
    )
    
    args = parser.parse_args()
    main(args)