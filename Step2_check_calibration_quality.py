import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
base_path = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(base_path, "Output", "Step1_Calibrate_Gabillon_model_parameters", "Gabillon_calibration_results.csv")
futures_file = os.path.join(base_path, "Input_data", "wti_futures_bbg_historical_constant_maturity.csv")

output_dir = os.path.join(base_path, "Output", "Step2_Check_Gabillon_calibration_quality")
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
params_df = pd.read_csv(input_file, parse_dates=["Date"])
futures_df = pd.read_csv(futures_file, parse_dates=["Date"])

# Only subset maturities
subset_cols = ["1M", "3M", "6M", "12M"]
maturity_map = {"1M": 1/12, "3M": 3/12, "6M": 6/12, "12M": 12/12}

# -----------------------------
# Gabillon forward function
# -----------------------------
def gabillon_forward(T, kappa_X, kappa_Y, X0, Y0):
    BX = np.exp(-kappa_X * T)
    BY = np.exp(-kappa_Y * T)
    return X0 * BX + Y0 * BY

# -----------------------------
# Step 1: Compute RMSE per day
# -----------------------------
rmse_results = []
for _, row in params_df.iterrows():
    d = row["Date"]
    subdf = futures_df[futures_df["Date"] == d][subset_cols].iloc[0]

    kappa_X, kappa_Y, X0, Y0 = row["kappa_X"], row["kappa_Y"], row["X0"], row["Y0"]

    errors = []
    for col in subset_cols:
        T = maturity_map[col]
        F_model = np.exp(gabillon_forward(T, kappa_X, kappa_Y, X0, Y0))
        F_mkt = subdf[col]
        errors.append((F_mkt - F_model)**2)
    rmse = np.sqrt(np.mean(errors))
    rmse_results.append({"Date": d, "RMSE": rmse})

rmse_df = pd.DataFrame(rmse_results)
rmse_df.to_csv(os.path.join(output_dir, "Gabillon_calibration_RMSE.csv"), index=False)

# -----------------------------
# Step 2: Plot parameter time series
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(params_df["Date"], params_df["kappa_X"], label="kappa_X")
plt.plot(params_df["Date"], params_df["kappa_Y"], label="kappa_Y")
plt.legend()
plt.title("Gabillon Parameters over Time")
plt.savefig(os.path.join(output_dir, "Gabillon_parameters.png"))
plt.close()

# -----------------------------
# Step 3: Sample curve fits
# -----------------------------
sample_dates = params_df["Date"].iloc[::250]  # pick 1 every ~250 days
for d in sample_dates:
    row = params_df[params_df["Date"] == d].iloc[0]
    kappa_X, kappa_Y, X0, Y0 = row["kappa_X"], row["kappa_Y"], row["X0"], row["Y0"]

    mkt = futures_df[futures_df["Date"] == d][subset_cols].iloc[0]
    maturities = np.linspace(1/12, 1, 50)
    fitted = [np.exp(gabillon_forward(T, kappa_X, kappa_Y, X0, Y0)) for T in maturities]

    plt.figure(figsize=(6,4))
    plt.plot([maturity_map[c] for c in subset_cols], mkt.values, "o", label="Market")
    plt.plot(maturities, fitted, "-", label="Gabillon Fit")
    plt.title(f"Curve Fit on {d.date()}")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"Gabillon_fit_{d.date()}.png"))
    plt.close()
