import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize

base_path = r"G:\My Drive\1_Career\Working Note\Related_Codes\7_Gabillon_calibration"
output_dir = os.path.join(base_path, "Output", "Step1_Calibrate_Gabillon_model_parameters")
os.makedirs(output_dir, exist_ok=True)

constant_maturity_curve_df = pd.read_csv(r"G:\My Drive\1_Career\Working Note\Related_Codes\7_Gabillon_calibration\Input_data\wti_futures_bbg_historical_constant_maturity.csv")

print(constant_maturity_curve_df)
df = constant_maturity_curve_df
df["Date"] = pd.to_datetime(df["Date"])

# Step 1: Select subset maturities (1M, 3M, 6M, 12M)
subset_cols = ["1M", "3M", "6M", "12M"]
maturities = np.array([1/12, 3/12, 6/12, 12/12]) # years
# Reshape into long format
long_df = df.melt(id_vars = "Date", 
                  value_vars = subset_cols,
                  var_name = "Maturity",
                  value_name = "Price")
# Map maturity labels to year fractions
maturity_map = {"1M": 1/12, "3M": 3/12, "6M": 6/12, "12M": 12/12}
long_df["Maturity"] = long_df["Maturity"].map(maturity_map)
print(long_df)

# Step 2: Gabillon forward log-price function
def gabillon_forward(T, kappa_X, kappa_Y, X0, Y0):
    BX = np.exp(-kappa_X * T)
    BY = np.exp(-kappa_Y * T)
    return X0 * BX + Y0 * BY

# Step 3: Calibration routine for one date
def calibrate_for_date(subdf):
    def objective(params):
        kappa_X, kappa_Y, X0, Y0 = params
        errors = []
        for T, Fmkt in zip(subdf["Maturity"], subdf["Price"]):
            F_model = np.exp(gabillon_forward(T, kappa_X, kappa_Y, X0, Y0))
            errors.append((Fmkt - F_model)**2)
        return np.sum(errors)

    # Initial guess
    logS0 = np.log(subdf["Price"].iloc[0])
    x0 = [1.0, 0.2, 0.0, logS0]
    bounds = [(0.01, 5), (0.01, 1), (-2,2), (logS0-1, logS0+1)]

# Step 4: loop over all dates
# Step 4: Loop over all dates and store calibration results
results = []
for d, subdf in long_df.groupby("Date"):
    res = minimize(
        lambda params: np.sum([
            (Fmkt - np.exp(gabillon_forward(T, *params)))**2
            for T, Fmkt in zip(subdf["Maturity"], subdf["Price"])
        ]),
        x0=[1.0, 0.2, 0.0, np.log(subdf["Price"].iloc[0])],
        bounds=[(0.01,5), (0.01,1), (-2,2), 
                (np.log(subdf["Price"].iloc[0])-1, np.log(subdf["Price"].iloc[0])+1)],
        method="L-BFGS-B"
    )
    kappa_X, kappa_Y, X0, Y0 = res.x
    results.append({
        "Date": d, "kappa_X": kappa_X, "kappa_Y": kappa_Y, "X0": X0, "Y0": Y0
    })

output_file = os.path.join(output_dir, "Gabillon_calibration_results.csv")
param_df = pd.DataFrame(results)
param_df.to_csv(output_file, index=False)

print(param_df.head())




     
