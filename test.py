import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================
# 1. 读数据
# =====================
features = pd.read_csv("./data/train_features.csv")
labels   = pd.read_csv("./data/train_labels.csv")

# 你代码里是 iloc[:, 3:]，这里保持一致
X = features.iloc[:, 3:]
Y = labels.copy()

# =====================
# 2. 找字段名
# =====================
in_spectra_cols = [c for c in X.columns if c.startswith("EDFA_input_spectra_")]
wss_cols        = [c for c in X.columns if c.startswith("DUT_WSS_activated_channel_index_")]
gain_cols       = [c for c in Y.columns if c.startswith("calculated_gain_spectra_")]

assert len(in_spectra_cols) == len(wss_cols) == len(gain_cols)

# global output power
out_total = X["EDFA_output_power_total"].values  # shape (B,)

# =====================
# 3. 转成 numpy
# =====================
P_in   = X[in_spectra_cols].values        # (B, N)
G      = Y[gain_cols].values              # (B, N)
WSS    = X[wss_cols].values                # (B, N)

# =====================
# 4. 物理重构 output total
# =====================
# dB → linear
P_out_linear = WSS * np.power(10.0, (P_in + G) / 10.0)

# sum over channels
P_out_linear_sum = np.sum(P_out_linear, axis=1)

# linear → dB（避免 log(0)）
P_out_recon = 10.0 * np.log10(P_out_linear_sum + 1e-12)

# =====================
# 5. 对比 & 误差统计
# =====================
error = P_out_recon - out_total

print("===== Output power consistency check =====")
print(f"Mean error (dB): {np.mean(error):.3f}")
print(f"Std  error (dB): {np.std(error):.3f}")
print(f"95% |error| <= {np.percentile(np.abs(error), 95):.3f} dB")

# =====================
# 6. 可视化
# =====================
plt.figure(figsize=(5,5))
plt.scatter(out_total, P_out_recon, s=5, alpha=0.5)
lims = [
    min(out_total.min(), P_out_recon.min()),
    max(out_total.max(), P_out_recon.max()),
]
plt.plot(lims, lims, "k--")
plt.xlabel("EDFA_output_power_total (CSV)")
plt.ylabel("Reconstructed output power")
plt.title("Physical consistency check")
plt.grid(True)
plt.show()

plt.figure()
plt.hist(error, bins=50)
plt.xlabel("Reconstruction error (dB)")
plt.title("Output power reconstruction error")
plt.grid(True)
plt.show()
