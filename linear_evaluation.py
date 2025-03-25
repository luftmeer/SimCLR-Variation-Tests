import argparse
import yaml

import torch
import torch.nn as nn

from utils.dataset_loader import get_dataset
from models.encoder import get_encoder
from simclr.simclr import SimCLR
from utils.loader import load_model, save_evaluation, save_model_eval
from torcheval.metrics import MulticlassAccuracy
from utils.log_loss import log_evaluation
import time

from tqdm import tqdm

def train(simclr_model, model, optimizer, criterion, train_loader, device, args):
    accuracy_epoch = MulticlassAccuracy(num_classes=args.n_classes)
    loss_epoch = 0
    for step, (img, target) in tqdm(enumerate(train_loader), desc='LE Training', total=len(train_loader)):
        optimizer.zero_grad()
        
        img = img.to(device)
        target = target.to(device)
        
        h, _ = simclr_model([img])
        
        out = model(h[0])
        loss = criterion(out, target)
        
        accuracy_epoch.update(out, target)
        loss_epoch += loss.item()
        
        loss.backward()
        optimizer.step()
    return loss_epoch / len(train_loader), accuracy_epoch.compute()

def test(simclr_model, model, criterion, test_loader, device, args):
    loss_epoch = 0
    accuracy_epoch = MulticlassAccuracy(num_classes=args.n_classes)
    model.eval()
    for step, (img, target) in tqdm(enumerate(test_loader), desc='Evaluating:', total=len(test_loader)):
        model.zero_grad()
        
        img = img.to(device)
        target = target.to(device)
        
        h, _ = simclr_model([img])
        
        out = model(h[0])
        loss = criterion(out, target)
        
        accuracy_epoch.update(out, target)
        loss_epoch += loss.item()
        
    return loss_epoch / len(test_loader), accuracy_epoch.compute()
        

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset retrieval
    train_dataset = get_dataset(dataset_name=args.dataset_name, train=True, image_size=args.resize, HF_TOKEN=args.HF_TOKEN, eval=True, args=args)
    test_dataset = get_dataset(dataset_name=args.dataset_name, train=False, image_size=args.resize, HF_TOKEN=args.HF_TOKEN, eval=True, args=args)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Encoder
    encoder, n_features = get_encoder(encoder=args.encoder, widening=args.widening)
    
    # Model
    simclr_model = SimCLR(encoder=encoder, n_features=n_features, projection_dim=args.projection_dim, image_size=args.resize, batch_size=args.batch_size, device=device).to(device)
    simclr_model_state, cpt_epoch = load_model(path=args.checkpoint, device=device, eval=True)
    simclr_model.load_state_dict(simclr_model_state)
    simclr_model.eval() # Freeze Model
    
    # Classifier 
    model = nn.Linear(n_features, args.n_classes)
    model = model.to(device)
    model.train()
    
    # Optimizer & Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Train
    best_accuracy = 0.
    for epoch in range(args.epochs_le):
        start = time.time()
        loss_epoch, accuracy_epoch = train(simclr_model, model, optimizer, criterion, train_loader, device, args)
        end = time.time()
        
        print(f"Epoch {epoch+1} | Loss: {loss_epoch} | Accuracy: {accuracy_epoch}")
        
        # General save after n-epochs
        if (epoch+1) %args.save_every_epoch_le == 0:
            save_model_eval(simclr_model=simclr_model, model=model, args=args, cpt_epoch=cpt_epoch, epoch=epoch, optimizer=optimizer)
        
        # Current best model save point
        if accuracy_epoch > best_accuracy:
            save_model_eval(simclr_model=simclr_model, model=model, args=args, cpt_epoch=cpt_epoch, epoch=epoch, optimizer=optimizer, best_model=True)
            best_accuracy = accuracy_epoch
        
        log_evaluation(epoch=epoch, loss=loss_epoch, accuracy=accuracy_epoch, args=args, elapsed_time=end-start)
        
    # Evaluate
    start = time.time()
    loss_epoch, accuracy_epoch = test(simclr_model, model, criterion, test_loader, device, args)
    end = time.time()
    
    print(f"[EVAL]\t Loss: {loss_epoch} | Accuracy: {accuracy_epoch}")        
    log_evaluation(epoch=epoch+1, loss=loss_epoch, accuracy=accuracy_epoch, args=args, elapsed_time=end-start)
    
    
    # Save Evaluation
    print("Save final model")
    save_evaluation(simclr_model=simclr_model, model=model, args=args, cpt_epoch=cpt_epoch, epoch=epoch+1)
    
    return print("Finished...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Linear Evaluation of SimCLR",
    )
    
    parser.add_argument('--checkpoint', type=str)
    
    parser.add_argument('--config', type=str, default='./config/le/default.yaml')
    
    parser.add_argument('--slurm_job_id', type=int, default=None)
    
    # Parse arguments known up till here, the rest via config file
    args = parser.parse_known_args()[0]
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        for elem in config:
            k, v = elem.popitem()
            parser.add_argument(f"--{k}", default=v, type=type(v))    

    args = parser.parse_args()
    
    main(args)