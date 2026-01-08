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
# Set seeds
print("PyTorch version:", torch.__version__)
torch.manual_seed(256)
np.random.seed(256)



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

    args = parser.parse_args()
    
    X_train = pd.read_csv(TRAIN_FEATURE_PATH).iloc[:, 3:]
    y_train = pd.read_csv(TRAIN_LABEL_PATH)

    y_train.fillna(0, inplace=True)



    TrainModelName = model_prepath+args.save_model_name

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
    else:
        available_models = ["BasicFNN", "LinearGateNet", "GatedBasicFNN", "ResidualFNN", 
                           "AttentionFNN", "ChannelWiseFNN", "LightweightFNN", "HybridFNN", "DeepResidualFNN"]
        raise ValueError(f"Invalid nn_type: {args.nn_type}. Available: {', '.join(available_models)}")

    optimizer = optim.Adam(base_model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(args.epochs):
        # Train
        base_model.train()
        running_loss = 0.0
        total_size = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = base_model(xb)
            loss, loaded_size = custom_loss_L2_pytorch(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * loaded_size
            total_size += loaded_size
        # print(f"Total size: {total_size}")
        train_losses.append(running_loss / total_size)
        
        # Validate
        base_model.eval()
        val_loss = 0.0
        total_size = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = base_model(xb)
                loss, loaded_size = custom_loss_L2_pytorch(pred, yb)
                val_loss += loss.item() * loaded_size
                total_size += loaded_size
        val_losses.append(val_loss / total_size)
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: TrainLoss={train_losses[-1]:.5f}  ValLoss={val_losses[-1]:.5f}")
        if args.save_best:
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_model = copy.deepcopy(base_model)

    if not args.save_best:    
        best_model = base_model # point to the model after the last epoch
        print(f"Saving {args.nn_type} model after the last epoch")
    else:
        print(f"Saving {args.nn_type} model when the validation loss is the best (val_loss={best_val_loss:.5f})")
    
    torch.save(best_model, TrainModelName)
    
    # %%
    if args.plot_loss:
        plot_loss(1, train_losses, val_losses, 15)
    else:
        print("Plotting loss is disabled")

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
    output_path = f"./submission/my_submission_{args.nn_type}.csv"
    y_pred.to_csv(output_path, index=False)

