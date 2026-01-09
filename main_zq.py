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



#%%
# Full paths to training data
TRAIN_FEATURE_PATH = f"./data/train_features.csv"
TRAIN_LABEL_PATH   = f"./data/train_labels.csv"

# Full path to test data
TEST_FEATURE_PATH = f"./data/test_features.csv"

# Output folders
figure_prepath = f"./figures/"
model_prepath = f"./model/"


# %%
### helper labels - fixed
# Numchannels = 95
# num_inputFeatures = Numchannels * 2 + 4 # target_gain, target_gain_tilt, EDFA_input_power_total, EDFA_output_power_total
# labels = {"gainValue":'target_gain',
#           "EDFA_input":'EDFA_input_power_total',
#           "EDFA_output":'EDFA_output_power_total',
#           "inSpectra":'EDFA_input_spectra_',
#           "WSS":'DUT_WSS_activated_channel_index_',
#           "result":'calculated_gain_spectra_'}
# inSpectra_labels = [labels['inSpectra']+str(i).zfill(2) for i in range(0,Numchannels)]
# onehot_labels = [labels['WSS']+str(i).zfill(2) for i in range(0,Numchannels)]
# result_labels = [labels['result']+str(i).zfill(2) for i in range(0,Numchannels)]
# preProcess_labels = [labels['EDFA_input'],labels['EDFA_output']]
# preProcess_labels.extend(inSpectra_labels)


def custom_loss_L2_pytorch(y_pred, y_actual):
    # y_pred, y_actual shape: [batch, channels]
    # turn unloaded y_pred prediction to zero
    # For each value, if actual==0, pred->0; else pred unchanged
    y_pred_cast_unloaded_to_zero = torch.where(y_actual != 0, y_pred, torch.zeros_like(y_pred))
    error = (y_pred_cast_unloaded_to_zero - y_actual) ** 2
    loaded_size = (y_actual != 0).sum().float()
    # avoid division by zero
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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--nn_type", type=str, default="BasicFNN")
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--save_best", action="store_true", default=False) # default save the model after the last epoch, if save_best is True, save the model when the validation loss is the best

    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    X_train = pd.read_csv(TRAIN_FEATURE_PATH).iloc[:, 3:]
    y_train = pd.read_csv(TRAIN_LABEL_PATH)

    y_train.fillna(0, inplace=True)

    Numchannels = 95
    common_suffix = f"_{args.nn_type}_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}_seed{args.seed}"
    TrainModelName = "./model/" + args.nn_type + common_suffix + ".pt"

    # --- Torch: Prepare Data ---
    X_np = X_train.values.astype(np.float32)
    y_np = y_train.values.astype(np.float32)

    X_tensor = torch.from_numpy(X_np)
    y_tensor = torch.from_numpy(y_np)


    # Create TensorDataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Split into train and val (as keras validation_split=0.2)
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
        global_noise_std=0.0,)
    elif args.nn_type == "MymodelAttention":
        base_model = Mymodel(
            global_dim=4,
            numchannel=95,
            hidden_embed_dim=2,
            num_layers=3,
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

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    for epoch in range(args.epochs):
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

    if not args.save_best:    
        best_model = base_model # point to the model after the last epoch
        print(f"Saving {args.nn_type} model after the last epoch to {TrainModelName}")
    else:
        print(f"Saving {args.nn_type} model when the validation loss is the best (val_loss={best_val_loss:.5f}) to {TrainModelName}")
    
    torch.save(best_model, TrainModelName)
    
    # %%
    plot_loss(1, train_losses, val_losses, 15)

    # %%
    X_test_full = pd.read_csv(TEST_FEATURE_PATH)
    
    X_test = X_test_full.iloc[:, 5:]
    
    
    # print(f"X_tensor.shape: {X_tensor.shape}")
    # print(f"y_tensor.shape: {y_tensor.shape}")
    # print(f"X_test.shape: {X_test.shape}")
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

    # Save to CSV (mask same as before)
    mask = X_test[wss_cols].values == 1

    y_pred = pd.DataFrame(np.where(mask, y_pred.values, np.nan),
        columns=label_cols
    )
    y_pred.fillna(0, inplace=True)

    kaggle_ID = X_test_full.columns[0]
    y_pred.insert(0, kaggle_ID, X_test_full[kaggle_ID].values)

    # Save predictions
    output_path = f"./submission/my_submission_{common_suffix}.csv"
    y_pred.to_csv(output_path, index=False)
