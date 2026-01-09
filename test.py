import pandas as pd
import numpy as np

# Load the submissions
df_basic = pd.read_csv("./submission/my_submission_BasicFNN.csv")
df_complicated = pd.read_csv("./submission/my_submission_ComplicatedFNN.csv")

# Extract prediction values as numpy arrays (excluding the ID column)
id_col = df_basic.columns[0]
arr_basic = df_basic.drop(columns=[id_col]).values
arr_complicated = df_complicated.drop(columns=[id_col]).values

# Compute the difference array
diff = arr_basic - arr_complicated

# Show various difference metrics
print("Maximum absolute difference:", np.max(np.abs(diff)))
print("Mean absolute difference:", np.mean(np.abs(diff)))
print("Median absolute difference:", np.median(np.abs(diff)))
print("Standard deviation of difference:", np.std(diff))
print("Minimum difference:", np.min(diff))
print("Maximum difference:", np.max(diff))
