import numpy as np
import pandas as pd
import os

# -----------------------------
# Paths
# -----------------------------
base_path = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(base_path, "Output", "Step1_Calibrate_Gabillon_model_parameters", "Gabillon_calibration_results.csv")
output_dir = os.path.join(base_path, "Output", "Step3a_Synthetic_Option_Vols")
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Load Step1 calibration results
# -----------------------------
param_df = pd.read_csv(input_file, parse_dates=["Date"])

# -----------------------------
# Define maturities and base vols
# -----------------------------
maturities = np.array([1/12, 3/12, 6/12, 12/12])  # in years
base_vols = np.array([0.45, 0.40, 0.35, 0.30])    # typical CL smile (shorter tenor = higher vol)

# -----------------------------
# Construct synthetic vol surface
# -----------------------------
np.random.seed(42)
vol_data = []
for d in param_df["Date"]:
    # Random shocks (2 vol points stdev)
    shocks = np.random.normal(0, 0.02, size=len(base_vols))

    # Regime effect (COVID 2020, Ukraine war 2022)
    regime_shift = 0.10 if d.year == 2020 else (0.05 if d.year == 2022 else 0.0)

    # Generate today's vol curve
    vols_today = np.clip(base_vols + shocks + regime_shift, 0.1, 1.0)

    # Save each maturity point
    for T, v in zip(maturities, vols_today):
        vol_data.append({"Date": d, "Maturity": T, "ImpliedVol": v})

# -----------------------------
# Save results
# -----------------------------
vol_df = pd.DataFrame(vol_data)
output_file = os.path.join(output_dir, "Synthetic_option_vols.csv")
vol_df.to_csv(output_file, index=False)

print("Synthetic option volatilities saved to:", output_file)
print(vol_df.head())

