import argparse
import yaml

import torch
import torch.nn as nn

from utils.dataset_loader import get_dataset
from models.encoder import get_encoder
from simclr.simclr import SimCLR
from utils.loader import load_model, save_evaluation, save_model_eval
from torcheval.metrics import MulticlassAccuracy, MulticlassConfusionMatrix
from utils.log_loss import log_evaluation
import time
from utils.logger import LinearEvaluationMonitor

from tqdm import tqdm
import os

def train(simclr_model, model, optimizer, criterion, train_loader, device, args):
    top1 = MulticlassAccuracy(num_classes=args.n_classes)
    top5 = MulticlassAccuracy(k=5)
    cm = MulticlassConfusionMatrix(num_classes=args.n_classes)
    acc_per_class = MulticlassAccuracy(average=None, num_classes=args.n_classes)
    all_features = []
    all_labels = []
    loss_epoch = 0
    for step, (img, target) in tqdm(enumerate(train_loader), desc='LE Training', total=len(train_loader)):
        optimizer.zero_grad()
        
        img = img.to(device)
        target = target.to(device)
        with torch.no_grad():
            h, _ = simclr_model([img])
        
        all_features.append(h[0].cpu())
        all_labels.append(target.cpu())
        
        out = model(h[0])
        loss = criterion(out, target)
        
        top1.update(out, target)
        top5.update(out, target)
        cm.update(out, target)
        acc_per_class.update(out, target)
        loss_epoch += loss.item()
        
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()} | Total Loss: {loss_epoch} | Top-1: {top1.compute().item()} | Top-5: {top5.compute().item()} | Learning Rate: {optimizer.param_groups[0]['lr']}", flush=True)
    
    print(f"Step [{step}/{len(train_loader)}]\t Total Loss: {loss_epoch} | Top-1: {top1.compute().item()} | Top-5: {top5.compute().item()} | Learning Rate: {optimizer.param_groups[0]['lr']}", flush=True)
    features = torch.cat(all_features)
    labels = torch.cat(all_labels)
    
    return loss_epoch / len(train_loader), top1.compute().item(), top5.compute().item(), cm.compute(), acc_per_class.compute(), features, labels

def test(simclr_model, model, criterion, test_loader, device, args):
    top1 = MulticlassAccuracy(num_classes=args.n_classes)
    top5 = MulticlassAccuracy(k=5)
    cm = MulticlassConfusionMatrix(num_classes=args.n_classes)
    acc_per_class = MulticlassAccuracy(average=None, num_classes=args.n_classes)
    all_features = []
    all_labels = []
    loss_epoch = 0
    model.eval()
    for step, (img, target) in tqdm(enumerate(test_loader), desc='Evaluating:', total=len(test_loader)):
        model.zero_grad()
        
        img = img.to(device)
        target = target.to(device)
        
        h, _ = simclr_model([img])
        
        all_features.append(h[0].cpu())
        all_labels.append(target.cpu())
        
        out = model(h[0])
        loss = criterion(out, target)
        
        top1.update(out, target)
        top5.update(out, target)
        cm.update(out, target)
        acc_per_class.update(out, target)
        loss_epoch += loss.item()
        
        if step % 25 == 0:
            print(f"Step [{step}/{len(test_loader)}]\t Loss: {loss.item()} | Total Loss: {loss_epoch} | Top-1: {top1.compute().item()} | Top-5: {top5.compute().item()}", flush=True)
    
    print(f"Step [{step}/{len(test_loader)}]\t | Total Loss: {loss_epoch} | Top-1: {top1.compute().item()} | Top-5: {top5.compute().item()}", flush=True)
    features = torch.cat(all_features)
    labels = torch.cat(all_labels)
    
    return loss_epoch / len(test_loader), top1.compute().item(), top5.compute().item(), cm.compute(), acc_per_class.compute(), features, labels
        

def main(args):
    # Randomness
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Pre-Load old args
    old_args = load_model(args.checkpoint, device, args_only=True)
    args.encoder = old_args.encoder
    args.widening = old_args.widening
    
    
    # Dataset retrieval
    train_dataset = get_dataset(dataset_name=args.dataset_name, train=True, image_size=args.resize, HF_TOKEN=args.HF_TOKEN, eval=True, args=args)
    test_dataset = get_dataset(dataset_name=args.dataset_name, train=False, image_size=args.resize, HF_TOKEN=args.HF_TOKEN, eval=True, args=args)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Encoder
    encoder, n_features = get_encoder(encoder=args.encoder, widening=args.widening)
    
    # SimCLR Model
    simclr_model = SimCLR(encoder=encoder, n_features=n_features, projection_dim=args.projection_dim, image_size=args.resize, batch_size=args.batch_size, device=device).to(device)
    simclr_model_state, cpt_epoch = load_model(path=args.checkpoint, device=device, eval=True)
    simclr_model.load_state_dict(simclr_model_state)
    simclr_model.eval() # Freeze Model
    for param in simclr_model.parameters():
        param.requires_grad = False
    
    simclr_model = simclr_model.to(device)
    
    # Classifier 
    model = nn.Linear(n_features, args.n_classes)
    model = model.to(device)
    model.train()
    
    # Optimizer & Criterion
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Linear Evaluation Monitor
    monitor = LinearEvaluationMonitor(save_dir=f'./runs/{args.slurm_job_id}', class_names=train_dataset.classes)
    
    # Train
    best_accuracy = 0.
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1} of {args.epochs}', flush=True)
        start = time.time()
        loss_epoch, top1, top5, cm, acc_per_class, features, labels = train(simclr_model, model, optimizer, criterion, train_loader, device, args)
        end = time.time()
        
        print(f"[Epoch {epoch+1}] Loss: {loss_epoch} | Top-1: {top1} | Top-5: {top5} | Learning Rate: {optimizer.param_groups[0]['lr']}")
        
        # General save after n-epochs
        if (epoch+1) %args.save_every_epoch == 0:
            print(f"Save model at epoch {epoch+1}", flush=True)
            save_model_eval(simclr_model=simclr_model, model=model, args=args, cpt_epoch=cpt_epoch, epoch=epoch, optimizer=optimizer, base_folder='runs')
        
        # Current best model save point
        if top1 > best_accuracy:
            print(f'Save new best model at epoch {epoch+1} with top-1 accuracy of {top1}', flush=True)
            save_model_eval(simclr_model=simclr_model, model=model, args=args, cpt_epoch=cpt_epoch, epoch=epoch, optimizer=optimizer, best_model=True, base_folder='runs')
            best_accuracy = top1
        
        log_evaluation(epoch=epoch, loss=loss_epoch, accuracy=top1, args=args, elapsed_time=end-start, cpt_epoch=cpt_epoch, top5=top5, base_folder='runs')
        print(f'Execute Monitor Logging', flush=True)
        monitor.log_metrics(epoch=epoch, 
                            top1=top1, 
                            top5=top5, 
                            loss=loss_epoch, 
                            lr=optimizer.param_groups[0]['lr'], 
                            eval_time=end-start, 
                            per_class_acc=acc_per_class, 
                            model=model, 
                            features=features, 
                            labels=labels)
        monitor.log_confusion_matrix(cm_tensor=cm, epoch=epoch)
        
        scheduler.step()
        
    # Evaluate
    start = time.time()
    loss_epoch, top1, top5, cm, acc_per_class, features, labels = test(simclr_model, model, criterion, test_loader, device, args)
    end = time.time()
    
    print(f"[EVAL]\t Loss: {loss_epoch} | Top-1: {top1} | Top-5: {top5}", flush=True)        
    log_evaluation(epoch=epoch+1, loss=loss_epoch, accuracy=top1, args=args, elapsed_time=end-start, cpt_epoch=cpt_epoch, top5=top5, base_folder='runs')
    monitor.log_metrics(epoch=epoch+1, 
                            top1=top1, 
                            top5=top5, 
                            loss=loss_epoch, 
                            lr=optimizer.param_groups[0]['lr'], 
                            eval_time=end-start, 
                            per_class_acc=acc_per_class, 
                            model=model, 
                            features=features, 
                            labels=labels)
    monitor.log_confusion_matrix(cm_tensor=cm, epoch=epoch)
    
    # Save Evaluation
    print("Save final model", flush=True)
    save_evaluation(simclr_model=simclr_model, model=model, args=args, cpt_epoch=cpt_epoch, epoch=epoch+1, base_folder='runs')
    
    return print("Finished...", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Linear Evaluation of SimCLR",
    )
    
    parser.add_argument('--checkpoint', type=str)
    
    parser.add_argument('--config', type=str, default='./config/linear_evaluation_default.yaml')
    
    parser.add_argument('--slurm_job_id', type=int, default=None)
    
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