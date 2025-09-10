import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize

# -----------------------------
# Paths
# -----------------------------
base_path = r"G:\My Drive\1_Career\Working Note\Related_Codes\7_Gabillon_calibration"

params_file = os.path.join(base_path, "Output", "Step1_Calibrate_Gabillon_model_parameters", "Gabillon_calibration_results.csv")
vols_file = os.path.join(base_path, "Output", "Step3a_Synthetic_Option_Vols", "Synthetic_option_vols.csv")

output_dir = os.path.join(base_path, "Output", "Step3_Calibrate_Gabillon_volatility_parameters")
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
param_df = pd.read_csv(params_file, parse_dates=["Date"])
vol_df = pd.read_csv(vols_file, parse_dates=["Date"])

# -----------------------------
# Gabillon forward variance
# -----------------------------
def gabillon_forward_var(T, kappa_X, kappa_Y, sigma_X, sigma_Y, rho):
    term1 = (sigma_X**2 / (2*kappa_X)) * (1 - np.exp(-2*kappa_X*T))
    term2 = (sigma_Y**2 / (2*kappa_Y)) * (1 - np.exp(-2*kappa_Y*T))
    term3 = (2*rho*sigma_X*sigma_Y / (kappa_X + kappa_Y)) * (1 - np.exp(-(kappa_X+kappa_Y)*T))
    return term1 + term2 + term3

def gabillon_forward_vol(T, kappa_X, kappa_Y, sigma_X, sigma_Y, rho):
    return np.sqrt(gabillon_forward_var(T, kappa_X, kappa_Y, sigma_X, sigma_Y, rho) / T)

# -----------------------------
# Calibration routine (sigma_X, sigma_Y, rho)
# -----------------------------
def calibrate_vols(subdf, kappa_X, kappa_Y):
    maturities = subdf["Maturity"].values
    market_vols = subdf["ImpliedVol"].values

    def objective(params):
        sigma_X, sigma_Y, rho = params
        model_vols = [gabillon_forward_vol(T, kappa_X, kappa_Y, sigma_X, sigma_Y, rho) for T in maturities]
        return np.sum((np.array(model_vols) - market_vols)**2)

    x0 = [0.3, 0.1, 0.0]
    bounds = [(0.01, 2.0), (0.01, 1.0), (-0.99, 0.99)]
    res = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
    return res.x  # sigma_X, sigma_Y, rho

# -----------------------------
# Step 3: Loop over all dates
# -----------------------------
results = []
for d in param_df["Date"]:
    row = param_df[param_df["Date"] == d].iloc[0]
    kappa_X, kappa_Y = row["kappa_X"], row["kappa_Y"]

    subdf = vol_df[vol_df["Date"] == d]
    if subdf.empty:
        continue

    sigma_X, sigma_Y, rho = calibrate_vols(subdf, kappa_X, kappa_Y)
    results.append({"Date": d, "sigma_X": sigma_X, "sigma_Y": sigma_Y, "rho": rho})

# -----------------------------
# Save results
# -----------------------------
vol_param_df = pd.DataFrame(results)
output_file = os.path.join(output_dir, "Gabillon_volatility_calibration_results.csv")
vol_param_df.to_csv(output_file, index=False)

print(vol_param_df.head())
