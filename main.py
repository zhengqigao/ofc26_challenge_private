import argparse
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from utils.models import (
    AttentionFNN,
    BasicFNN,
    ChannelWiseFNN,
    DeepResidualFNN,
    GatedBasicFNN,
    HybridFNN,
    LightweightFNN,
    LinearGateNet,
    ResidualFNN,
    SpectralCNN,
    SpectralTransformer,
)

# Reproducibility
print("PyTorch version:", torch.__version__)
torch.manual_seed(256)
np.random.seed(256)

# Paths
TRAIN_FEATURE_PATH = "./data/train_features.csv"
TRAIN_LABEL_PATH = "./data/train_labels.csv"
TEST_FEATURE_PATH = "./data/test_features.csv"

# Output folders
FIGURE_DIR = "./figures"
MODEL_DIR = "./model"
SUBMISSION_DIR = "./submission"

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Feature labels
NUM_CHANNELS = 95
LABELS = {
    "gainValue": "target_gain",
    "EDFA_input": "EDFA_input_power_total",
    "EDFA_output": "EDFA_output_power_total",
    "inSpectra": "EDFA_input_spectra_",
    "WSS": "DUT_WSS_activated_channel_index_",
    "result": "calculated_gain_spectra_",
}


# -------------------------
# Losses and regularizers
# -------------------------

def custom_loss_L2_pytorch(y_pred, y_actual):
    y_pred_cast = torch.where(y_actual != 0, y_pred, torch.zeros_like(y_pred))
    error = (y_pred_cast - y_actual) ** 2
    loaded_size = (y_actual != 0).sum().float().item()
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
    loss = 0.5 * quadratic**2 + delta * linear
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


def compute_base_loss(y_pred, y_actual, loss_type, huber_delta):
    if loss_type == "masked_rmse":
        return custom_loss_L2_pytorch(y_pred, y_actual)
    if loss_type == "masked_huber":
        return masked_huber_loss(y_pred, y_actual, delta=huber_delta)
    raise ValueError(f"Unknown loss_type: {loss_type}")


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


# -------------------------
# Feature utilities
# -------------------------

def build_feature_columns(num_channels, labels):
    global_cols = [
        labels["gainValue"],
        "target_gain_tilt",
        labels["EDFA_input"],
        labels["EDFA_output"],
    ]
    spectra_cols = [labels["inSpectra"] + str(i).zfill(2) for i in range(num_channels)]
    wss_cols = [labels["WSS"] + str(i).zfill(2) for i in range(num_channels)]
    feature_cols = global_cols + [col for pair in zip(spectra_cols, wss_cols) for col in pair]
    return global_cols, spectra_cols, wss_cols, feature_cols


def select_feature_frame(df, feature_cols):
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing feature columns: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return df[feature_cols].copy()


def fit_feature_scalers(x_df, global_scale_cols, spectra_cols):
    global_scaler = None
    if global_scale_cols:
        global_scaler = StandardScaler()
        global_scaler.fit(x_df[global_scale_cols])
    spectra_scaler = StandardScaler()
    spectra_scaler.fit(x_df[spectra_cols].values / 100.0)
    return global_scaler, spectra_scaler


def apply_feature_scalers(x_df, global_scale_cols, spectra_cols, global_scaler, spectra_scaler):
    x_scaled = x_df.copy()
    if global_scaler is not None and global_scale_cols:
        x_scaled[global_scale_cols] = global_scaler.transform(x_scaled[global_scale_cols])
    scaled_spectra = spectra_scaler.transform(x_scaled[spectra_cols].values / 100.0)
    x_scaled[spectra_cols] = scaled_spectra * 100.0
    return x_scaled


# -------------------------
# Model utilities
# -------------------------

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
            num_channels,
            hidden_channels=args.cnn_channels,
            dropout=args.cnn_dropout,
            spectra_noise_std=args.spectra_noise_std,
            global_noise_std=args.global_noise_std,
        )
    available = [
        "BasicFNN",
        "LinearGateNet",
        "GatedBasicFNN",
        "ResidualFNN",
        "AttentionFNN",
        "ChannelWiseFNN",
        "LightweightFNN",
        "HybridFNN",
        "DeepResidualFNN",
        "SpectralTransformer",
        "SpectralCNN",
    ]
    raise ValueError(f"Invalid nn_type: {args.nn_type}. Available: {', '.join(available)}")


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
    device = next(base_model.parameters()).device

    for epoch in range(args.epochs):
        # Train
        base_model.train()
        running_loss = 0.0
        total_size = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
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
                xb = xb.to(device)
                yb = yb.to(device)
                pred = base_model(xb)
                base_loss, loaded_size = compute_base_loss(pred, yb, args.loss_type, args.huber_delta)
                if loaded_size == 0:
                    continue
                val_loss += base_loss.item() * loaded_size
                total_size += loaded_size
        val_loss = val_loss / max(total_size, 1)
        val_losses.append(val_loss)

        if epoch % 100 == 0:
            print(
                f"{prefix}Epoch {epoch+1}/{args.epochs}: "
                f"TrainLoss={train_losses[-1]:.5f}  ValLoss={val_losses[-1]:.5f}"
            )

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


def write_submission(y_pred_array, output_tag, y_train_columns, x_test_full, x_test_features, wss_cols):
    y_pred = pd.DataFrame(y_pred_array, columns=y_train_columns)
    mask = x_test_features[wss_cols].values == 1
    y_pred = pd.DataFrame(np.where(mask, y_pred.values, np.nan), columns=y_train_columns)
    y_pred.fillna(0, inplace=True)

    kaggle_id = x_test_full.columns[0]
    y_pred.insert(0, kaggle_id, x_test_full[kaggle_id].values)

    output_path = os.path.join(SUBMISSION_DIR, f"my_submission_{output_tag}.csv")
    y_pred.to_csv(output_path, index=False)
    return output_path


def plot_loss(indx, train_losses, val_losses, ignore_index):
    plt.figure(indx)
    plt.plot(train_losses[ignore_index:], label="loss")
    plt.plot(val_losses[ignore_index:], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Error [gain]")
    plt.legend()
    plt.grid(True)
    plt.show()


# -------------------------
# Defaults and validation
# -------------------------

def resolve_defaults(args):
    def default_if_none(value, default):
        return default if value is None else value

    is_spectral = args.nn_type in {"SpectralTransformer", "SpectralCNN"}

    if is_spectral:
        args.loss_type = default_if_none(args.loss_type, "masked_huber")
        args.smooth_lambda = default_if_none(args.smooth_lambda, 0.05)
        args.spectra_noise_std = default_if_none(args.spectra_noise_std, 0.01)
        args.global_noise_std = default_if_none(args.global_noise_std, 0.02)
        args.channel_dropout = default_if_none(args.channel_dropout, 0.1)
        args.weight_decay = default_if_none(args.weight_decay, 0.05)
        args.early_stop_patience = default_if_none(args.early_stop_patience, 100)
        args.early_stop_min_delta = default_if_none(args.early_stop_min_delta, 1e-4)
        args.lr_scheduler = default_if_none(args.lr_scheduler, "plateau")
        args.plateau_patience = default_if_none(args.plateau_patience, 50)
        args.plateau_factor = default_if_none(args.plateau_factor, 0.5)
        args.min_lr = default_if_none(args.min_lr, 1e-6)
        args.cv_folds = default_if_none(args.cv_folds, 5)
        args.spectral_embed_dim = default_if_none(args.spectral_embed_dim, 64)
        args.spectral_heads = default_if_none(args.spectral_heads, 4)
        args.spectral_layers = default_if_none(args.spectral_layers, 2)
        args.spectral_dropout = default_if_none(args.spectral_dropout, 0.25)
        args.spectral_ff_mult = default_if_none(args.spectral_ff_mult, 2)
        args.cnn_channels = default_if_none(args.cnn_channels, 32)
        args.cnn_dropout = default_if_none(args.cnn_dropout, 0.25)
        args.use_scaler = default_if_none(args.use_scaler, True)
    else:
        args.loss_type = default_if_none(args.loss_type, "masked_rmse")
        args.smooth_lambda = default_if_none(args.smooth_lambda, 0.0)
        args.spectra_noise_std = default_if_none(args.spectra_noise_std, 0.0)
        args.global_noise_std = default_if_none(args.global_noise_std, 0.0)
        args.channel_dropout = default_if_none(args.channel_dropout, 0.0)
        args.weight_decay = default_if_none(args.weight_decay, 0.01)
        args.early_stop_patience = default_if_none(args.early_stop_patience, 0)
        args.early_stop_min_delta = default_if_none(args.early_stop_min_delta, 0.0)
        args.lr_scheduler = default_if_none(args.lr_scheduler, "none")
        args.plateau_patience = default_if_none(args.plateau_patience, 0)
        args.plateau_factor = default_if_none(args.plateau_factor, 0.5)
        args.min_lr = default_if_none(args.min_lr, 1e-6)
        args.cv_folds = default_if_none(args.cv_folds, 1)
        args.spectral_embed_dim = default_if_none(args.spectral_embed_dim, 64)
        args.spectral_heads = default_if_none(args.spectral_heads, 4)
        args.spectral_layers = default_if_none(args.spectral_layers, 2)
        args.spectral_dropout = default_if_none(args.spectral_dropout, 0.1)
        args.spectral_ff_mult = default_if_none(args.spectral_ff_mult, 2)
        args.cnn_channels = default_if_none(args.cnn_channels, 32)
        args.cnn_dropout = default_if_none(args.cnn_dropout, 0.1)
        args.use_scaler = default_if_none(args.use_scaler, True)

    if args.loss_type not in {"masked_rmse", "masked_huber"}:
        raise ValueError("loss_type must be masked_rmse or masked_huber")
    if args.lr_scheduler not in {"none", "plateau"}:
        raise ValueError("lr_scheduler must be none or plateau")
    if args.cv_folds < 1:
        raise ValueError("cv_folds must be >= 1")

    if args.cv_group_col in {"none", "None", ""}:
        args.cv_group_col = None
    if args.cv_folds > 1 and args.cv_group_col is None:
        args.cv_group_col = "edfa_index"

    args.save_fold_csv = default_if_none(args.save_fold_csv, args.cv_folds > 1)

    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_model_name", type=str, default="model.pt")
    parser.add_argument("--nn_type", type=str, default="BasicFNN")
    parser.add_argument("--plot_loss", action="store_true", default=False)
    parser.add_argument("--save_best", action="store_true", default=False)

    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--loss_type", type=str, default=None)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--smooth_lambda", type=float, default=None)
    parser.add_argument("--spectra_noise_std", type=float, default=None)
    parser.add_argument("--global_noise_std", type=float, default=None)
    parser.add_argument("--channel_dropout", type=float, default=None)
    parser.add_argument("--use_scaler", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save_fold_csv", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--early_stop_min_delta", type=float, default=None)
    parser.add_argument("--lr_scheduler", type=str, default=None)
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

    parser.add_argument("--cnn_channels", type=int, default=None)
    parser.add_argument("--cnn_dropout", type=float, default=None)

    return parser.parse_args()


# -------------------------
# Main
# -------------------------

def main():
    args = resolve_defaults(parse_args())

    # Load data
    X_train_full = pd.read_csv(TRAIN_FEATURE_PATH)
    y_train = pd.read_csv(TRAIN_LABEL_PATH)
    y_train.fillna(0, inplace=True)

    global_cols, spectra_cols, wss_cols, feature_cols = build_feature_columns(NUM_CHANNELS, LABELS)
    X_train_features = select_feature_frame(X_train_full, feature_cols)

    X_test_full = pd.read_csv(TEST_FEATURE_PATH)
    X_test_features = select_feature_frame(X_test_full, feature_cols)

    # Scaling policy: keep target_gain/tilt raw for spectral baselines
    if args.nn_type in {"SpectralTransformer", "SpectralCNN"}:
        global_scale_cols = [LABELS["EDFA_input"], LABELS["EDFA_output"]]
    else:
        global_scale_cols = global_cols

    train_indices = np.arange(len(X_train_features))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_model_name = os.path.join(MODEL_DIR, args.save_model_name)

    if args.cv_folds > 1:
        if args.cv_group_col:
            if args.cv_group_col not in X_train_full.columns:
                raise ValueError(f"cv_group_col {args.cv_group_col} not found in training features.")
            groups = X_train_full[args.cv_group_col].values
            splitter = GroupKFold(n_splits=args.cv_folds)
            splits = splitter.split(train_indices, groups=groups)
        else:
            splitter = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.cv_seed)
            splits = splitter.split(train_indices)

        pred_sum = None
        fold_best_vals = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
            x_train_fold = X_train_features.iloc[train_idx]
            x_val_fold = X_train_features.iloc[val_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_val_fold = y_train.iloc[val_idx]

            if args.use_scaler:
                global_scaler, spectra_scaler = fit_feature_scalers(
                    x_train_fold, global_scale_cols, spectra_cols
                )
                x_train_fold = apply_feature_scalers(
                    x_train_fold, global_scale_cols, spectra_cols, global_scaler, spectra_scaler
                )
                x_val_fold = apply_feature_scalers(
                    x_val_fold, global_scale_cols, spectra_cols, global_scaler, spectra_scaler
                )
                x_test_fold = apply_feature_scalers(
                    X_test_features, global_scale_cols, spectra_cols, global_scaler, spectra_scaler
                )
            else:
                x_test_fold = X_test_features

            train_loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(x_train_fold.values.astype(np.float32)),
                    torch.from_numpy(y_train_fold.values.astype(np.float32)),
                ),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
            )
            val_loader = DataLoader(
                TensorDataset(
                    torch.from_numpy(x_val_fold.values.astype(np.float32)),
                    torch.from_numpy(y_val_fold.values.astype(np.float32)),
                ),
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
            )

            model = build_model(args, x_train_fold.shape[1], NUM_CHANNELS).to(device)
            best_model, train_losses, val_losses, best_val = train_model(
                model, train_loader, val_loader, args, fold_idx=fold_idx
            )
            fold_best_vals.append(best_val)

            fold_path = fold_model_path(train_model_name, fold_idx)
            torch.save(best_model, fold_path)

            x_test_tensor = torch.from_numpy(x_test_fold.values.astype(np.float32)).to(device)
            fold_pred = predict_model(best_model, x_test_tensor)
            pred_sum = fold_pred if pred_sum is None else pred_sum + fold_pred
            if args.save_fold_csv:
                fold_tag = f"{args.nn_type}_cv{args.cv_folds}_fold{fold_idx}"
                write_submission(
                    fold_pred,
                    fold_tag,
                    y_train.columns,
                    X_test_full,
                    X_test_features,
                    wss_cols,
                )

        y_pred_array = pred_sum / args.cv_folds
        print(f"CV best val mean={np.mean(fold_best_vals):.5f} std={np.std(fold_best_vals):.5f}")
        if args.plot_loss:
            print("Plotting loss is disabled for cross-validation.")
        output_tag = f"{args.nn_type}_cv{args.cv_folds}"
    else:
        n_total = len(X_train_features)
        n_val = int(0.2 * n_total)
        rng = np.random.default_rng(256)
        perm = rng.permutation(n_total)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        x_train = X_train_features.iloc[train_idx]
        x_val = X_train_features.iloc[val_idx]
        y_train_split = y_train.iloc[train_idx]
        y_val_split = y_train.iloc[val_idx]

        if args.use_scaler:
            global_scaler, spectra_scaler = fit_feature_scalers(
                x_train, global_scale_cols, spectra_cols
            )
            x_train = apply_feature_scalers(
                x_train, global_scale_cols, spectra_cols, global_scaler, spectra_scaler
            )
            x_val = apply_feature_scalers(
                x_val, global_scale_cols, spectra_cols, global_scaler, spectra_scaler
            )
            x_test_scaled = apply_feature_scalers(
                X_test_features, global_scale_cols, spectra_cols, global_scaler, spectra_scaler
            )
        else:
            x_test_scaled = X_test_features

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(x_train.values.astype(np.float32)),
                torch.from_numpy(y_train_split.values.astype(np.float32)),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(x_val.values.astype(np.float32)),
                torch.from_numpy(y_val_split.values.astype(np.float32)),
            ),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        model = build_model(args, x_train.shape[1], NUM_CHANNELS).to(device)
        best_model, train_losses, val_losses, best_val_loss = train_model(
            model, train_loader, val_loader, args
        )

        if args.save_best or args.early_stop_patience > 0:
            print(
                f"Saving {args.nn_type} model when the validation loss is the best "
                f"(val_loss={best_val_loss:.5f})"
            )
        else:
            print(f"Saving {args.nn_type} model after the last epoch")
        torch.save(best_model, train_model_name)

        if args.plot_loss:
            plot_loss(1, train_losses, val_losses, 15)
        else:
            print("Plotting loss is disabled")

        x_test_tensor = torch.from_numpy(x_test_scaled.values.astype(np.float32)).to(device)
        y_pred_array = predict_model(best_model, x_test_tensor)
        output_tag = args.nn_type

    write_submission(
        y_pred_array,
        output_tag,
        y_train.columns,
        X_test_full,
        X_test_features,
        wss_cols,
    )


if __name__ == "__main__":
    main()
