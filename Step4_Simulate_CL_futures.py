import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Paths
# -----------------------------
base_path = r"G:\My Drive\1_Career\Working Note\Related_Codes\7_Gabillon_calibration"
curve_file = os.path.join(base_path, "Output", "Step1_Calibrate_Gabillon_model_parameters", "Gabillon_calibration_results.csv")
vol_file = os.path.join(base_path, "Output", "Step3_Calibrate_Gabillon_volatility_parameters", "Gabillon_volatility_calibration_results.csv")

output_dir = os.path.join(base_path, "Output", "Step4_Simulate_CL_Futures")
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Load calibration results
# -----------------------------
curve_df = pd.read_csv(curve_file, parse_dates=["Date"])
vol_df = pd.read_csv(vol_file, parse_dates=["Date"])
full_df = pd.merge(curve_df, vol_df, on="Date", how="inner")

# -----------------------------
# Simulation function
# -----------------------------
def simulate_gabillon(date, T=1.0, dt=1/252, M=5000, n_paths_plot=20):
    """Simulate CL spot and futures using Gabillon model for a given date"""
    row = full_df[full_df["Date"] == date].iloc[0]
    kappa_X, kappa_Y = row["kappa_X"], row["kappa_Y"]
    sigma_X, sigma_Y, rho = row["sigma_X"], row["sigma_Y"], row["rho"]
    X0, Y0 = row["X0"], row["Y0"]
    S0 = np.exp(X0 + Y0)

    N = int(T/dt)
    cov = np.array([[1, rho],[rho, 1]])
    L = np.linalg.cholesky(cov)

    X = np.zeros((M, N+1))
    Y = np.zeros((M, N+1))
    S = np.zeros((M, N+1))
    X[:,0], Y[:,0], S[:,0] = X0, Y0, S0

    for t in range(N):
        dW = np.dot(L, np.random.randn(2, M)) * np.sqrt(dt)
        dX = -kappa_X * X[:,t] * dt + sigma_X * dW[0]
        dY = -kappa_Y * (Y[:,t]) * dt + sigma_Y * dW[1]
        X[:,t+1] = X[:,t] + dX
        Y[:,t+1] = Y[:,t] + dY
        S[:,t+1] = np.exp(X[:,t+1] + Y[:,t+1])

    # -----------------------------
    # Plot sample spot paths
    # -----------------------------
    plt.figure(figsize=(8,4))
    for i in range(n_paths_plot):
        plt.plot(S[i,:], lw=1)
    plt.title(f"CL Spot Price Simulation ({date.date()})")
    plt.xlabel("Days")
    plt.ylabel("Price (USD)")
    plt.savefig(os.path.join(output_dir, f"Spot_paths_{date.date()}.png"))
    plt.close()

    # -----------------------------
    # Futures distributions at custom expiries
    # -----------------------------
    expiries = {
        "10d": 10/252,
        "45d": 45/252,
        "6M": 6/12,
        "12M": 12/12,
    }

    print(f"\nSimulation complete for {date.date()} (S0={S0:.2f}). Futures Estimates:")
    for label, Tmat in expiries.items():
        idx = int(Tmat/dt)
        futures_paths = S[:, idx]
        futures_mean = futures_paths.mean()

        # Plot histogram
        plt.figure(figsize=(6,4))
        plt.hist(futures_paths, bins=50, density=True, alpha=0.7)
        plt.title(f"Simulated {label} Futures Distribution ({date.date()})")
        plt.xlabel("Price (USD)")
        plt.ylabel("Density")
        plt.savefig(os.path.join(output_dir, f"Futures_dist_{label}_{date.date()}.png"))
        plt.close()

        print(f"  {label}: {futures_mean:.2f} USD")

# -----------------------------
# Run Case Studies
# -----------------------------
case_dates = [
    pd.Timestamp("2020-03-16"), # COVID crash
    pd.Timestamp("2022-03-08"), # Ukraine war spike
    pd.Timestamp("2025-05-28"), # Latest available
]

for d in case_dates:
    simulate_gabillon(d)
