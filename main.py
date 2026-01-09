# %%
# libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math, os, shutil
# from prettytable import PrettyTable
import scipy.stats as stats
from sklearn.model_selection import train_test_split, KFold, GroupKFold
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
import copy
from utils.models import *
from utils.helper import seed_everything
# Set seeds
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

# Create output folders if not existed
os.makedirs(figure_prepath, exist_ok=True)
os.makedirs(model_prepath, exist_ok=True)

# Plotting configuration
figureFrontSize = 12
figureName_post = "_test.png"


# %%
### helper labels - fixed
Numchannels = 95
num_inputFeatures = Numchannels * 2 + 4 # target_gain, target_gain_tilt, EDFA_input_power_total, EDFA_output_power_total
labels = {"gainValue":'target_gain',
          "EDFA_input":'EDFA_input_power_total',
          "EDFA_output":'EDFA_output_power_total',
          "inSpectra":'EDFA_input_spectra_',
          "WSS":'DUT_WSS_activated_channel_index_',
          "result":'calculated_gain_spectra_'}
inSpectra_labels = [labels['inSpectra']+str(i).zfill(2) for i in range(0,Numchannels)]
onehot_labels = [labels['WSS']+str(i).zfill(2) for i in range(0,Numchannels)]
result_labels = [labels['result']+str(i).zfill(2) for i in range(0,Numchannels)]
preProcess_labels = [labels['EDFA_input'],labels['EDFA_output']]
preProcess_labels.extend(inSpectra_labels)

# %%
# def dB_to_linear(data):
#   return np.power(10,data/10)

# def linear_TO_Db(data):
#   result = 10*np.log10(data).to_numpy()
#   return result[result != -np.inf]

# def linear_TO_Db_full(data):
#   result = 10*np.log10(data).to_numpy()
#   result[result == -np.inf] = 0
#   return result

# def divideZero(numerator,denominator):
#   with np.errstate(divide='ignore'):
#     result = numerator / denominator
#     result[denominator == 0] = 0
#   return result

# %%
### PyTorch Loss
def custom_loss_L2_pytorch(y_pred, y_actual):
    # y_pred, y_actual shape: [batch, channels]
    # turn unloaded y_pred prediction to zero
    # For each value, if actual==0, pred->0; else pred unchanged
    y_pred_cast_unloaded_to_zero = torch.where(y_actual != 0, y_pred, torch.zeros_like(y_pred))
    error = (y_pred_cast_unloaded_to_zero - y_actual) ** 2
    loaded_size = (y_actual != 0).sum().float().item()
    # avoid division by zero
    loss = torch.sqrt(error.sum() / (loaded_size + 1e-8))
    return loss, loaded_size


def masked_huber_loss(y_pred, y_actual, delta=1.0):
    mask = y_actual != 0
    loaded_size = mask.sum().float().item()
    if loaded_size == 0:
        return torch.zeros((), device=y_pred.device), 0
    diff = y_pred - y_actual
    abs_diff = diff.abs()
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    loss = loss[mask].mean()
    return loss, loaded_size


def smoothness_loss(y_pred, y_actual):
    mask = y_actual != 0
    if mask.sum().item() == 0:
        return torch.zeros((), device=y_pred.device)
    diff = y_pred[:, 1:] - y_pred[:, :-1]
    mask_adj = (mask[:, 1:] & mask[:, :-1]).float()
    denom = mask_adj.sum().clamp_min(1.0)
    return (diff.abs() * mask_adj).sum() / denom


def apply_channel_dropout(xb, yb, drop_prob):
    if drop_prob <= 0:
        return xb, yb
    wss = xb[:, 5::2]
    if wss.numel() == 0:
        return xb, yb
    drop_mask = (torch.rand_like(wss) < drop_prob) & (wss > 0)
    if not drop_mask.any():
        return xb, yb
    xb = xb.clone()
    yb = yb.clone()
    spectra = xb[:, 4::2]
    spectra = torch.where(drop_mask, torch.zeros_like(spectra), spectra)
    wss = torch.where(drop_mask, torch.zeros_like(wss), wss)
    xb[:, 4::2] = spectra
    xb[:, 5::2] = wss
    yb = torch.where(drop_mask, torch.zeros_like(yb), yb)
    return xb, yb


def compute_base_loss(y_pred, y_actual, loss_type, huber_delta):
    if loss_type == "masked_rmse":
        return custom_loss_L2_pytorch(y_pred, y_actual)
    if loss_type == "masked_huber":
        return masked_huber_loss(y_pred, y_actual, delta=huber_delta)
    raise ValueError(f"Unknown loss_type: {loss_type}")


def build_model(args, input_dim, num_channels):
    if args.nn_type == "BasicFNN":
        return BasicFNN(input_dim, num_channels)
    if args.nn_type == "LinearGateNet":
        return LinearGateNet()
    if args.nn_type == "GatedBasicFNN":
        return GatedBasicFNN()
    if args.nn_type == "ResidualFNN":
        return ResidualFNN(input_dim, num_channels)
    if args.nn_type == "AttentionFNN":
        return AttentionFNN(input_dim, num_channels)
    if args.nn_type == "ChannelWiseFNN":
        return ChannelWiseFNN(input_dim, num_channels)
    if args.nn_type == "LightweightFNN":
        return LightweightFNN(input_dim, num_channels)
    if args.nn_type == "HybridFNN":
        return HybridFNN(input_dim, num_channels)
    if args.nn_type == "DeepResidualFNN":
        return DeepResidualFNN(input_dim, num_channels)
    if args.nn_type == "SpectralTransformer":
        return SpectralTransformer(
            input_dim,
            num_channels,
            embed_dim=args.spectral_embed_dim,
            num_heads=args.spectral_heads,
            num_layers=args.spectral_layers,
            dropout=args.spectral_dropout,
            ff_mult=args.spectral_ff_mult,
            spectra_noise_std=args.spectra_noise_std,
            global_noise_std=args.global_noise_std,
        )
    if args.nn_type == "SpectralCNN":
        return SpectralCNN(
        input_dim,
        output_dim = num_channels,
        num_channels = num_channels,
        global_dim=4,
        hidden_channels=32,
        dropout=args.spectral_dropout,
        spectra_noise_std=args.spectra_noise_std,
        global_noise_std=args.global_noise_std,)
        
    available_models = ["BasicFNN", "LinearGateNet", "GatedBasicFNN", "ResidualFNN",
                        "AttentionFNN", "ChannelWiseFNN", "LightweightFNN", "HybridFNN",
                        "DeepResidualFNN", "SpectralTransformer", "SpectralCNN"]
    raise ValueError(f"Invalid nn_type: {args.nn_type}. Available: {', '.join(available_models)}")


def train_model(base_model, train_loader, val_loader, args, fold_idx=None):
    optimizer = optim.AdamW(base_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.min_lr,
        )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model = None
    epochs_no_improve = 0
    track_best = args.save_best or (args.early_stop_patience > 0)
    prefix = f"[Fold {fold_idx}] " if fold_idx is not None else ""

    for epoch in range(args.epochs):
        # Train
        base_model.train()
        running_loss = 0.0
        total_size = 0
        for xb, yb in train_loader:
            xb = xb.to(next(base_model.parameters()).device)
            yb = yb.to(next(base_model.parameters()).device)
            if args.channel_dropout > 0:
                xb, yb = apply_channel_dropout(xb, yb, args.channel_dropout)
            optimizer.zero_grad()
            pred = base_model(xb)
            base_loss, loaded_size = compute_base_loss(pred, yb, args.loss_type, args.huber_delta)
            if loaded_size == 0:
                continue
            loss = base_loss
            if args.smooth_lambda > 0:
                loss = loss + args.smooth_lambda * smoothness_loss(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += base_loss.item() * loaded_size
            total_size += loaded_size
        train_losses.append(running_loss / max(total_size, 1))

        # Validate
        base_model.eval()
        val_loss = 0.0
        total_size = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(next(base_model.parameters()).device)
                yb = yb.to(next(base_model.parameters()).device)
                pred = base_model(xb)
                base_loss, loaded_size = compute_base_loss(pred, yb, args.loss_type, args.huber_delta)
                if loaded_size == 0:
                    continue
                val_loss += base_loss.item() * loaded_size
                total_size += loaded_size
        val_loss = val_loss / max(total_size, 1)
        val_losses.append(val_loss)

        if epoch % 100 == 0:
            print(f"{prefix}Epoch {epoch+1}/{args.epochs}: TrainLoss={train_losses[-1]:.5f}  ValLoss={val_losses[-1]:.5f}")

        if val_loss < best_val_loss - args.early_stop_min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if track_best:
                best_model = copy.deepcopy(base_model)
        else:
            epochs_no_improve += 1

        if scheduler is not None:
            scheduler.step(val_loss)

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(f"{prefix}Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.5f})")
            break

    final_model = best_model if (track_best and best_model is not None) else base_model
    return final_model, train_losses, val_losses, best_val_loss


def predict_model(model, x_tensor):
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(x_tensor)
    return y_pred_tensor.detach().cpu().numpy()


def fold_model_path(base_path, fold_idx):
    if base_path.endswith(".pt"):
        return f"{base_path[:-3]}_fold{fold_idx}.pt"
    return f"{base_path}_fold{fold_idx}.pt"



def plot_loss(indx,train_losses,val_losses,ingnoreIndex):
    plt.figure(indx)
    plt.plot(train_losses[ingnoreIndex:], label='loss')
    plt.plot(val_losses[ingnoreIndex:], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [gain]')
    plt.legend()
    plt.grid(True)
    plt.show()

# def figure_comp(figIndx,y_test_result,y_pred_result,filename,setFrontSize):
#     plt.figure(figIndx)
#     plt.axes(aspect='equal')
#     plt.scatter(y_test_result, y_pred_result)
#     plt.xlabel('Measured EDFA Gain (dB)', fontsize=setFrontSize)
#     plt.ylabel('predicted EDFA Gain (dB)', fontsize=setFrontSize)
#     minAxis = math.floor(min(y_test_result.min(),y_pred_result.min()) - 0.5)
#     maxAxis = math.ceil (max(y_test_result.max(),y_pred_result.max()) + 0.5)
#     limss = [*np.arange(minAxis,maxAxis+1,1)]
#     lims = [limss[0],limss[-1]]
#     plt.xlim(lims)
#     plt.ylim(lims)
#     plt.plot(lims, lims, 'k--')
#     plt.xticks(ticks=limss,labels=limss,fontsize=setFrontSize)
#     plt.yticks(fontsize=setFrontSize)
#     plt.savefig(figure_prepath+filename, dpi=900)

# def figure_hist(figIndx,error,filename,setFrontSize):
#     plt.figure(figIndx)
#     bins_list = [*np.arange(-0.6,0.7,0.1)]
#     labelList = ['-0.6','','-0.4','','-0.2','','0.0','','0.2','','0.4','','0.6']
#     plt.hist(error, bins=bins_list)
#     for i in [-0.2,-0.1,0.1,0.2]: # helper vertical line
#         plt.axvline(x=i,color='black',ls='--')
#     plt.xlabel('Prediction Gain Error (dB)', fontsize=setFrontSize)
#     plt.ylabel('Histogram', fontsize=setFrontSize)
#     plt.xticks(ticks=bins_list, labels=labelList, fontsize=setFrontSize)
#     plt.yticks(fontsize=setFrontSize)
#     plt.savefig(figure_prepath+filename, dpi=900)

# def plot_per_channel_error(y_pred,y_test):
#     y_pred_result = linear_TO_Db_full(y_pred)
#     y_test_result = linear_TO_Db_full(y_test)
#     error = y_test_result - y_pred_result
#     error_min_0_1s,error_means,error_min_0_2s,within95ranges,mses = [],[],[],[],[]
#     for j in range(len(error[0])):
#         error_channel = error[:,j]
#         error_channel = error_channel[error_channel!=0]
#         # calculate the distribution
#         error_reasonable = [i for i in error_channel if abs(i)<=0.2]
#         error_measureError = [i for i in error_channel if abs(i)<=0.1]
#         error_min_0_1 = len(error_measureError)/len(error_channel)
#         error_min_0_2 = len(error_reasonable)/len(error_channel)
#         error_sorted = np.sort(abs(error_channel))
#         within95range = error_sorted[int(0.95*len(error_channel))]
#         mse = (np.square(error_channel)).mean(axis=None)
#         error_mean = error_channel.mean(axis=None)
#         error_means.append(error_mean)
#         error_min_0_1s.append(error_min_0_1)
#         error_min_0_2s.append(error_min_0_2)
#         within95ranges.append(within95range)
#         mses.append(mse)
#     plt.figure(103)
#     plt.plot(mses)
#     plt.xlabel('Channel indices')
#     plt.ylabel('MSE (dB^2)')
#     plt.title("Per channel MSE")
      
# def getErrorInfo(error):
#     error_reasonable = [i for i in error if abs(i)<=0.2]
#     error_measureError = [i for i in error if abs(i)<=0.1]
#     error_min_0_1 = len(error_measureError)/len(error)
#     error_min_0_2 = len(error_reasonable)/len(error)
#     error_sorted = np.sort(abs(error))
#     within95range = error_sorted[int(0.95*len(error))]
#     mse = (np.square(error)).mean(axis=None)
#     return_error_min_0_1 = "{:.2f}".format(error_min_0_1)
#     return_error_min_0_2 = "{:.2f}".format(error_min_0_2)
#     return_within95range = "{:.2f}".format(within95range)
#     return_mse = "{:.2f}".format(mse)
#     return return_error_min_0_1,return_error_min_0_2,return_within95range,return_mse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_model_name", type=str, default="model.pt")
    parser.add_argument("--nn_type", type=str, default="BasicFNN")
    parser.add_argument("--plot_loss", action="store_true", default=False)
    parser.add_argument("--save_best", action="store_true", default=False) # default save the model after the last epoch, if save_best is True, save the model when the validation loss is the best
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--loss_type", type=str, default=None)  # masked_rmse or masked_huber
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--smooth_lambda", type=float, default=None)
    parser.add_argument("--spectra_noise_std", type=float, default=None)
    parser.add_argument("--global_noise_std", type=float, default=None)
    parser.add_argument("--channel_dropout", type=float, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--early_stop_min_delta", type=float, default=None)
    parser.add_argument("--lr_scheduler", type=str, default=None)  # none or plateau
    parser.add_argument("--plateau_patience", type=int, default=None)
    parser.add_argument("--plateau_factor", type=float, default=None)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--cv_folds", type=int, default=None)
    parser.add_argument("--cv_group_col", type=str, default=None)
    parser.add_argument("--cv_seed", type=int, default=256)
    parser.add_argument("--spectral_embed_dim", type=int, default=None)
    parser.add_argument("--spectral_heads", type=int, default=None)
    parser.add_argument("--spectral_layers", type=int, default=None)
    parser.add_argument("--spectral_dropout", type=float, default=None)
    parser.add_argument("--spectral_ff_mult", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    def _default_if_none(value, default):
        return default if value is None else value

    if args.nn_type == "SpectralTransformer":
        args.loss_type = _default_if_none(args.loss_type, "masked_huber")
        args.smooth_lambda = _default_if_none(args.smooth_lambda, 0.05)
        args.spectra_noise_std = _default_if_none(args.spectra_noise_std, 0.01)
        args.global_noise_std = _default_if_none(args.global_noise_std, 0.02)
        args.channel_dropout = _default_if_none(args.channel_dropout, 0.1)
        args.weight_decay = _default_if_none(args.weight_decay, 0.05)
        args.early_stop_patience = _default_if_none(args.early_stop_patience, 150)
        args.early_stop_min_delta = _default_if_none(args.early_stop_min_delta, 1e-4)
        args.lr_scheduler = _default_if_none(args.lr_scheduler, "plateau")
        args.plateau_patience = _default_if_none(args.plateau_patience, 50)
        args.plateau_factor = _default_if_none(args.plateau_factor, 0.5)
        args.min_lr = _default_if_none(args.min_lr, 1e-6)
        args.cv_folds = _default_if_none(args.cv_folds, 1)
        args.spectral_embed_dim = _default_if_none(args.spectral_embed_dim, 64)
        args.spectral_heads = _default_if_none(args.spectral_heads, 4)
        args.spectral_layers = _default_if_none(args.spectral_layers, 2)
        args.spectral_dropout = _default_if_none(args.spectral_dropout, 0.25)
        args.spectral_ff_mult = _default_if_none(args.spectral_ff_mult, 2)
    else:
        args.loss_type = _default_if_none(args.loss_type, "masked_rmse")
        args.smooth_lambda = _default_if_none(args.smooth_lambda, 0.0)
        args.spectra_noise_std = _default_if_none(args.spectra_noise_std, 0.0)
        args.global_noise_std = _default_if_none(args.global_noise_std, 0.0)
        args.channel_dropout = _default_if_none(args.channel_dropout, 0.0)
        args.weight_decay = _default_if_none(args.weight_decay, 0.01)
        args.early_stop_patience = _default_if_none(args.early_stop_patience, 0)
        args.early_stop_min_delta = _default_if_none(args.early_stop_min_delta, 0.0)
        args.lr_scheduler = _default_if_none(args.lr_scheduler, "none")
        args.plateau_patience = _default_if_none(args.plateau_patience, 0)
        args.plateau_factor = _default_if_none(args.plateau_factor, 0.5)
        args.min_lr = _default_if_none(args.min_lr, 1e-6)
        args.cv_folds = _default_if_none(args.cv_folds, 1)
        args.spectral_embed_dim = _default_if_none(args.spectral_embed_dim, 64)
        args.spectral_heads = _default_if_none(args.spectral_heads, 4)
        args.spectral_layers = _default_if_none(args.spectral_layers, 2)
        args.spectral_dropout = _default_if_none(args.spectral_dropout, 0.1)
        args.spectral_ff_mult = _default_if_none(args.spectral_ff_mult, 2)

    if args.loss_type not in {"masked_rmse", "masked_huber"}:
        raise ValueError(f"Invalid loss_type: {args.loss_type}. Use masked_rmse or masked_huber.")
    if args.lr_scheduler not in {"none", "plateau"}:
        raise ValueError(f"Invalid lr_scheduler: {args.lr_scheduler}. Use none or plateau.")
    if args.cv_folds < 1:
        raise ValueError("--cv_folds must be >= 1.")
    
    X_train_full = pd.read_csv(TRAIN_FEATURE_PATH)
    X_train = X_train_full.iloc[:, 3:]
    y_train = pd.read_csv(TRAIN_LABEL_PATH)

    y_train.fillna(0, inplace=True)

    TrainModelName = model_prepath + args.save_model_name

    # --- Torch: Prepare Data ---
    X_np = X_train.values.astype(np.float32)
    y_np = y_train.values.astype(np.float32)

    X_tensor = torch.from_numpy(X_np)
    y_tensor = torch.from_numpy(y_np)

    # Create TensorDataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Test data ---
    X_test_full = pd.read_csv(TEST_FEATURE_PATH)
    X_test = X_test_full.iloc[:, 5:]
    X_test_tensor = torch.from_numpy(X_test.values.astype(np.float32)).to(device)

    if args.cv_folds > 1:
        indices = np.arange(len(dataset))
        if args.cv_group_col:
            if args.cv_group_col not in X_train_full.columns:
                raise ValueError(f"cv_group_col {args.cv_group_col} not found in training features.")
            groups = X_train_full[args.cv_group_col].values
            splitter = GroupKFold(n_splits=args.cv_folds)
            splits = splitter.split(indices, groups=groups)
        else:
            splitter = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.cv_seed)
            splits = splitter.split(indices)

        pred_sum = None
        fold_best_vals = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
            train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, drop_last=False)
            val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, drop_last=False)

            base_model = build_model(args, X_tensor.shape[1], Numchannels).to(device)
            best_model, train_losses, val_losses, best_val = train_model(
                base_model, train_loader, val_loader, args, fold_idx=fold_idx
            )
            fold_best_vals.append(best_val)

            fold_path = fold_model_path(TrainModelName, fold_idx)
            torch.save(best_model, fold_path)

            fold_pred = predict_model(best_model, X_test_tensor)
            pred_sum = fold_pred if pred_sum is None else pred_sum + fold_pred

        y_pred_array = pred_sum / args.cv_folds
        print(f"CV best val mean={np.mean(fold_best_vals):.5f} std={np.std(fold_best_vals):.5f}")
        if args.plot_loss:
            print("Plotting loss is disabled for cross-validation.")
        output_tag = f"{args.nn_type}_cv{args.cv_folds}"
    else:
        # Split into train and val (as keras validation_split=0.2)
        n_total = len(dataset)
        n_val = int(0.2 * n_total)
        n_train = n_total - n_val
        train_dataset, val_dataset = random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(256)
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

        base_model = build_model(args, X_tensor.shape[1], Numchannels).to(device)
        best_model, train_losses, val_losses, best_val_loss = train_model(base_model, train_loader, val_loader, args)

        if args.save_best or args.early_stop_patience > 0:
            print(f"Saving {args.nn_type} model when the validation loss is the best (val_loss={best_val_loss:.5f})")
        else:
            print(f"Saving {args.nn_type} model after the last epoch")
        torch.save(best_model, TrainModelName)

        if args.plot_loss:
            plot_loss(1, train_losses, val_losses, 15)
        else:
            print("Plotting loss is disabled")

        y_pred_array = predict_model(best_model, X_test_tensor)
        output_tag = args.nn_type
    
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
    output_path = f"./submission/my_submission_{output_tag}.csv"
    y_pred.to_csv(output_path, index=False)
