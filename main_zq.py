# %%
# libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math,os,shutil
from prettytable import PrettyTable
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import copy
from utils.models import *
import random

print("PyTorch version:", torch.__version__)

# %%
# Utility function: load from checkpoint
def load_checkpoint(model, checkpoint_path, map_location=None):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif hasattr(model, 'load_state_dict'):
        model.load_state_dict(checkpoint)
    else:
        # may be whole-model serialization (discouraged, but support if so)
        model = checkpoint
    return model

# Full paths to training data
def load_csvs(paths):
    if isinstance(paths, (list, tuple)):
        return pd.concat([pd.read_csv(p) for p in paths], axis=0, ignore_index=True)
    else:
        return pd.read_csv(paths)

TRAIN_FEATURE_PATH = [
    "./data/train_features.csv",
    "./data/COSMOS_features.csv"
]
TRAIN_LABEL_PATH = [
    "./data/train_labels.csv",
    "./data/COSMOS_labels.csv"
]

FINAL_TRAIN_FEATURE_PATH = [
    "./data/train_features.csv",
]
FINAL_TRAIN_LABEL_PATH = [
    "./data/train_labels.csv",
]

TEST_FEATURE_PATH = f"./data/test_features.csv"

figure_prepath = f"./figures/"
model_prepath = f"./model/"

def custom_loss_L2_pytorch(y_pred, y_actual):
    y_pred_cast_unloaded_to_zero = torch.where(y_actual != 0, y_pred, torch.zeros_like(y_pred))
    error = (y_pred_cast_unloaded_to_zero - y_actual) ** 2
    loaded_size = (y_actual != 0).sum().float()
    loss = torch.sqrt(error.sum() / (loaded_size + 1e-8))
    return loss

def plot_loss(indx,train_losses,val_losses,ingnoreIndex):
    plt.figure(indx)
    plt.plot(train_losses[ingnoreIndex:], label='loss')
    plt.plot(val_losses[ingnoreIndex:], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [gain]')
    plt.legend()
    plt.grid(True)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)

def get_scheduler(optimizer, scheduler_type, epochs, **kwargs):
    scheduler_type = scheduler_type.lower()
    if scheduler_type == "none":
        return None
    elif scheduler_type == "step":
        step_size = kwargs.get("step_size", 100)
        gamma = kwargs.get("gamma", 0.5)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "cosine":
        T_max = kwargs.get("t_max", epochs)
        eta_min = kwargs.get("eta_min", 0)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        raise ValueError(f"Unsupported lr_scheduler: {scheduler_type}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--nn_type", type=str, default="BasicFNN")
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--save_best", action="store_true", default=False)
    parser.add_argument("--hidden_embed_dim", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--lr_scheduler", type=str, default="none", choices=["none","step","cosine"], 
                        help="选择lr scheduler策略：none/step/cosine (可扩展类型)")
    parser.add_argument("--scheduler_step_size", type=int, default=100, 
                        help="StepLR: 每多少epoch衰减")
    parser.add_argument("--scheduler_gamma", type=float, default=0.5, 
                        help="StepLR: 衰减系数")
    parser.add_argument("--scheduler_eta_min", type=float, default=0,
                        help="CosineAnnealingLR: 最小lr")
    parser.add_argument("--cosmos_ratio", type=float, default=1.0, help="COSMOS数据扩增倍数，相对于原始train的样本数")
    # 新增: 是否加载checkpoint继续训练
    parser.add_argument("--resume_from", type=str, default=None, help="传入checkpoint的模型路径，继续训练")
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    # 读取原始训练数据
    train_features_df = pd.read_csv("./data/train_features.csv")
    train_labels_df = pd.read_csv("./data/train_labels.csv")
    cosmos_features_df = pd.read_csv("./data/COSMOS_features.csv")
    cosmos_labels_df = pd.read_csv("./data/COSMOS_labels.csv")

    num_train_rows = len(train_features_df)
    cosmos_use_num = int(args.cosmos_ratio * num_train_rows)
    if cosmos_use_num > 0 and len(cosmos_features_df) > 0:
        replace_flag = cosmos_use_num > len(cosmos_features_df)
        cosmos_features_used = cosmos_features_df.sample(n=cosmos_use_num, replace=replace_flag, random_state=args.seed)
        cosmos_labels_used = cosmos_labels_df.loc[cosmos_features_used.index].reset_index(drop=True)
    else:
        cosmos_features_used = cosmos_features_df.iloc[[]]
        cosmos_labels_used = cosmos_labels_df.iloc[[]]

    all_features = pd.concat([train_features_df, cosmos_features_used.reset_index(drop=True)], axis=0, ignore_index=True)
    all_labels = pd.concat([train_labels_df, cosmos_labels_used], axis=0, ignore_index=True)

    X_train = all_features
    y_train = all_labels
    print(f"use number of samples: {len(X_train)}")
    
    X_train = X_train.iloc[:, 3:]
    y_train.fillna(0, inplace=True)

    Numchannels = 95
    
    common_suffix = f"_{args.nn_type}_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}_seed{args.seed}_hidden_embed_dim{args.hidden_embed_dim}_hidden_dim{args.hidden_dim}_num_layers{args.num_layers}_lr_scheduler{args.lr_scheduler}_scheduler_step_size{args.scheduler_step_size}_scheduler_gamma{args.scheduler_gamma}_scheduler_eta_min{args.scheduler_eta_min}_cosmos_ratio{args.cosmos_ratio}"
    TrainModelName = "./model/model_" + args.nn_type + ".pt"

    # --- Torch: Prepare Data ---
    X_np = X_train.values.astype(np.float32)
    y_np = y_train.values.astype(np.float32)
    X_tensor = torch.from_numpy(X_np)
    y_tensor = torch.from_numpy(y_np)

    dataset = TensorDataset(X_tensor, y_tensor)
    n_total = len(dataset)
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(256))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # --- Model create/train ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.nn_type == "BasicFNN":
        base_model = BasicFNN(X_tensor.shape[1], Numchannels).to(device)
    elif args.nn_type == "LinearGateNet":
        base_model = LinearGateNet().to(device)
    elif args.nn_type == "GatedBasicFNN":
        base_model = GatedBasicFNN().to(device)
    elif args.nn_type == "ResidualFNN":
        base_model = ResidualFNN(X_tensor.shape[1], Numchannels).to(device)
    elif args.nn_type == "AttentionFNN":
        base_model = AttentionFNN(X_tensor.shape[1], Numchannels).to(device)
    elif args.nn_type == "ChannelWiseFNN":
        base_model = ChannelWiseFNN(X_tensor.shape[1], Numchannels).to(device)
    elif args.nn_type == "LightweightFNN":
        base_model = LightweightFNN(X_tensor.shape[1], Numchannels).to(device)
    elif args.nn_type == "HybridFNN":
        base_model = HybridFNN(X_tensor.shape[1], Numchannels).to(device)
    elif args.nn_type == "DeepResidualFNN":
        base_model = DeepResidualFNN(X_tensor.shape[1], Numchannels).to(device)
    elif args.nn_type == "SpectralTransformer":
        base_model = SpectralTransformer(
            X_tensor.shape[1],
            Numchannels,
            num_channels=95,
            global_dim=4,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
            ff_mult=2,
            spectra_noise_std=0.0,
            global_noise_std=0.0,
        )
    elif args.nn_type == "MymodelAttention":
        base_model = Mymodel(
            global_dim=4,
            numchannel=95,
            hidden_embed_dim=args.hidden_embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            token_model="attention",
        ).to(device)
    elif args.nn_type == "MymodelConv":
        base_model = Mymodel(
            global_dim=4,
            numchannel=95,
            hidden_embed_dim=2,
            token_model="conv",
        ).to(device)
    elif args.nn_type == "SpectralCNN":
        base_model = SpectralCNN(
            X_tensor.shape[1],
            output_dim = 95,
            num_channels = 95,
            global_dim=4,
            hidden_channels=32,
            dropout=0.1,
            spectra_noise_std=0.0,
            global_noise_std=0.0,)
    elif args.nn_type == "ImprovedSpectralCNN":
        base_model = ImprovedSpectralCNN(
            X_tensor.shape[1],
            output_dim = 95,
            num_channels = 95,
            global_dim=4,
            hidden_channels=32,
            dropout=0.1,
            spectra_noise_std=0.0,
            global_noise_std=0.0,)
    elif args.nn_type == "ImprovedDeepSpectralCNN":
        base_model = ImprovedDeepSpectralCNN(
            X_tensor.shape[1],
            output_dim = 95,
            num_channels = 95,
            global_dim=4,
            hidden_channels=32,
            dropout=0.1,
            spectra_noise_std=0.0,
            global_noise_std=0.0,)
    elif args.nn_type == "ImprovedEmbedDeepSpectralCNN":
        base_model = ImprovedEmbedDeepSpectralCNN(
            X_tensor.shape[1],
            output_dim = 95,
            num_channels = 95,
            global_dim=4,
            hidden_channels=32,
            hidden_embed_dim=2,
            dropout=0.1,
            spectra_noise_std=0.0,
            global_noise_std=0.0,)
    elif args.nn_type == "ImprovedSpectralTransformer":
        base_model = ImprovedSpectralTransformer(
            input_dim=X_tensor.shape[1],
            output_dim=95,
            num_channels=95,
            global_dim=4,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.25,
            stochastic_depth_prob=0.1, 
            use_channel_weighting=True,
        ).to(device)
    else:
        raise ValueError(f"Invalid nn_type: {args.nn_type}.")

    base_model.to(device)
    num_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in model: {num_params/1e6:.2f} Million")

    optimizer = optim.AdamW(base_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler_kwargs = {}
    if args.lr_scheduler == "step":
        scheduler_kwargs = {'step_size': args.scheduler_step_size, 'gamma': args.scheduler_gamma}
    if args.lr_scheduler == "cosine":
        scheduler_kwargs = {'t_max': args.epochs, 'eta_min': args.scheduler_eta_min}
    lr_scheduler = get_scheduler(optimizer, args.lr_scheduler, args.epochs, **scheduler_kwargs)
    if lr_scheduler is not None:
        print(f"Using lr_scheduler: {args.lr_scheduler}")

    # ========== 加载已有checkpoint继续训练 ==========
    start_epoch = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None

    if args.resume_from is not None and os.path.isfile(args.resume_from):
        print(f"Loading checkpoint from {args.resume_from} to resume training.")
        checkpoint = torch.load(args.resume_from, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            base_model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'lr_scheduler_state_dict' in checkpoint and lr_scheduler is not None:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            if 'train_losses' in checkpoint: train_losses = checkpoint['train_losses']
            if 'val_losses' in checkpoint: val_losses = checkpoint['val_losses']
            if 'best_val_loss' in checkpoint: best_val_loss = checkpoint['best_val_loss']
            if 'epoch' in checkpoint: start_epoch = checkpoint['epoch'] + 1
            if 'best_model_state_dict' in checkpoint and args.save_best:
                best_model = copy.deepcopy(base_model)
                best_model.load_state_dict(checkpoint['best_model_state_dict'])
            else:
                best_model = copy.deepcopy(base_model)
        else:
            # whole model file, fallback to torch.save(model)
            base_model = checkpoint
            best_model = copy.deepcopy(base_model)
        print(f"Resumed from checkpoint at epoch={start_epoch}")
    else:
        if args.resume_from:
            print(f"Warning: Resume checkpoint path {args.resume_from} does not exist, training from scratch.")

    # Note: If loading optimizer/scheduler, learning rate may resume from checkpoint—overwrite if needed

    for epoch in range(start_epoch, args.epochs):
        # Train
        base_model.train()
        running_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = base_model(xb)
            loss = custom_loss_L2_pytorch(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        train_losses.append(running_loss / n_batches)
        # Validate
        base_model.eval()
        val_loss = 0.0
        n_batches_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = base_model(xb)
                loss = custom_loss_L2_pytorch(pred, yb)
                val_loss += loss.item()
                n_batches_val += 1
        val_losses.append(val_loss / n_batches_val)
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: TrainLoss={train_losses[-1]:.5f}  ValLoss={val_losses[-1]:.5f}")
        if args.save_best:
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_model = copy.deepcopy(base_model)
        if lr_scheduler is not None:
            lr_scheduler.step()
            if epoch % 100 == 0 or epoch == args.epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1} lr: {current_lr:.6f}")

        # 如果想每个epoch都保存checkpoint，可以在这里加保存代码
        # torch.save({...}, f'./model/latest_ckpt.pt')

    if not args.save_best:    
        best_model = base_model
        print(f"Saving {args.nn_type} model after the last epoch to {TrainModelName}")
    else:
        print(f"Saving {args.nn_type} model when the validation loss is the best (val_loss={best_val_loss:.5f}) to {TrainModelName}")

    torch.save(best_model, TrainModelName)

    # %% 保存完整checkpoint 方便resume
    # 保存训练过程信息：losses，state_dict等
    checkpoint_full = {
        'model_state_dict': base_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'epoch': epoch,
        'args': vars(args),
    }
    if args.save_best and best_model is not None:
        checkpoint_full['best_model_state_dict'] = best_model.state_dict()
    torch.save(checkpoint_full, TrainModelName.replace('.pt', '_ckpt.pt'))

    # %%
    plot_loss(1, train_losses, val_losses, 15)

    # %%
    X_test_full = pd.read_csv(TEST_FEATURE_PATH)
    X_test = X_test_full.iloc[:, 5:]

    # Predict on test set
    best_model.eval()
    X_test_np = X_test.values.astype(np.float32)
    X_test_tensor = torch.from_numpy(X_test_np).to(device)
    with torch.no_grad():
        y_pred_tensor = best_model(X_test_tensor)
        y_pred_array = y_pred_tensor.cpu().numpy()

    # Convert prediction to DataFrame
    y_pred = pd.DataFrame(y_pred_array, columns=y_train.columns)

    wss_cols = [col for col in X_test.columns if 'dut_wss_activated_channel_index' in col.lower()]
    label_cols = [col for col in y_train.columns if 'calculated_gain_spectra' in col.lower()]

    mask = X_test[wss_cols].values == 1

    y_pred = pd.DataFrame(np.where(mask, y_pred.values, np.nan),
        columns=label_cols
    )
    y_pred.fillna(0, inplace=True)

    kaggle_ID = X_test_full.columns[0]
    y_pred.insert(0, kaggle_ID, X_test_full[kaggle_ID].values)

    output_path = f"./submission/my_submission_{common_suffix}.csv"
    y_pred.to_csv(output_path, index=False)
