import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import collate
from dgllife.utils import EarlyStopping
from dl_model import Model
import warnings
warnings.filterwarnings("ignore")
import numpy  as np
import pickle
from sklearn.metrics import r2_score
import random

def evaluate(labels, preds):
    preds = preds.detach().cpu().numpy().reshape(-1)
    labels = labels.detach().cpu().numpy().reshape(-1)
    mse = np.mean((preds - labels) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - labels))
    r2 = r2_score(labels, preds)
    
    return rmse, mae, r2

def train(train_loader, val_loader, model, optimizer, scheduler, stopper):
    best_r2 = -1e9
    for epoch in range(500):
        model.train()
        res1=[]
        epoch_losses = [] 
        for i, batch1 in enumerate(train_loader):

            batch1 = batch1.to(device)

            labels = batch1.y
            model.zero_grad()

            pred = model(batch1.batch_size, batch1, batch1.fp)

            ts = labels.float()
            loss = model.label_loss(pred.view(-1), ts.view(-1))
            epoch_losses.append(loss.item()) 

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            rmse, mae, r2 = evaluate(labels, pred)
            res1.append([rmse, mae, r2])
        
        mean_all_loss = sum(epoch_losses) / len(epoch_losses) 
        train_results = pd.DataFrame(res1, columns=['RMSE', 'MAE', 'R2'])
        r1 = train_results.mean()
        print(
            f"epoch:{epoch}---train---RMSE:{r1['RMSE']:.4f}---MAE:{r1['MAE']:.4f}---R2:{r1['R2']:.4f}---Total Loss:{mean_all_loss:.4f}"
        )
        
        model.eval()
        res = []
        val_epoch_losses = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)  
                labels = batch.y
                batch_size = batch.batch_size
                fps_t = batch.fp
                pred = model(batch_size, batch, fps_t)
                ts = labels.float()
                loss = model.label_loss(pred.view(-1), ts.view(-1))
                val_epoch_losses.append(loss.item()) 

                rmse, mae, r2 = evaluate(labels, pred)
                res.append([rmse, mae, r2])
        
        val_mean_all_loss = sum(val_epoch_losses) / len(val_epoch_losses) 
        val_results = pd.DataFrame(res, columns=['RMSE', 'MAE', 'R2'])
        r = val_results.mean()
        if r['R2'] > best_r2:
            best_r2 = r['R2']
            torch.save(model.state_dict(), f"{output}/best_r2_model.pth")
        print(
            f"epoch:{epoch}---validation---RMSE:{r['RMSE']:.4f}---MAE:{r['MAE']:.4f}---R2:{r['R2']:.4f}---Total Loss:{val_mean_all_loss:.4f}---Best R2:{best_r2:.4f}"
        )

        scheduler.step(val_mean_all_loss)

        early_stop = stopper.step(val_mean_all_loss, model)
        if early_stop:
            break
    

def main(train_loader, val_loader, test_loader):
    model = Model(dropout=0.5, device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    stopper = EarlyStopping(mode='lower', filename=f'{output}/best_model.pkl', patience=30)
    train(train_loader, val_loader, model, optimizer, scheduler, stopper)
    stopper.load_checkpoint(model)
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    res = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            labels = batch.y
            batch_size = batch.batch_size
            fps_t = batch.fp
            pred = model(batch_size, batch, fps_t)
            total_preds = torch.cat((total_preds, pred.cpu()), 0)
            total_labels = torch.cat((total_labels, labels.cpu()), 0)

            rmse, mae, r2 = evaluate(labels, pred)
            res.append([rmse, mae, r2])

    test_results = pd.DataFrame(res, columns=['RMSE', 'MAE', 'R2'])
    r = test_results.mean()
    print("res:")
    print(f"test_---RMSE:{r['RMSE']:.4f}---MAE:{r['MAE']:.4f}---R2:{r['R2']:.4f}-")

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output = "./output/"

    def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    setup_seed(42)

    with open('./data/train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    
    with open('./data/valid_dataset.pkl', 'rb') as f:
        validate_dataset = pickle.load(f)

    with open('./data/test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)

    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate, drop_last=True, shuffle=True, num_workers=0)
    val_loader = DataLoader(validate_dataset, batch_size=128, collate_fn=collate, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=collate, drop_last=False)

    if not os.path.isdir(output):
       os.mkdir(output)
    main(train_loader, val_loader, test_loader)

        
    


