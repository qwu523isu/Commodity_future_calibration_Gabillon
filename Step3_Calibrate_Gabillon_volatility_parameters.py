import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize

# -----------------------------
# Gabillon forward variance
# -----------------------------
def int_exp(a, T):
    """Integral of exp(-a*u) over [0, T], including a=0."""
    T = np.asarray(T, dtype=float)
    if a == 0:
        return T
    return -np.expm1(-a * T) / a


def gabillon_forward_var(T, kappa_X, kappa_Y, sigma_X, sigma_Y, rho):
    term1 = sigma_X**2 * int_exp(2 * kappa_X, T)
    term2 = sigma_Y**2 * int_exp(2 * kappa_Y, T)
    term3 = 2 * rho * sigma_X * sigma_Y * int_exp(kappa_X + kappa_Y, T)
    return term1 + term2 + term3

def gabillon_forward_vol(T, kappa_X, kappa_Y, sigma_X, sigma_Y, rho):
    return np.sqrt(gabillon_forward_var(T, kappa_X, kappa_Y, sigma_X, sigma_Y, rho) / T)

# -----------------------------
# Bound diagnostics
# -----------------------------
PARAM_NAMES = ["sigma_X", "sigma_Y", "rho"]
BOUNDS = [(0.01, 2.0), (0.01, 1.0), (-0.99, 0.99)]

def bound_flags(x, bounds, tol=1e-6):
    """One bool per parameter: is it resting on either of its bounds?"""
    return [bool(abs(xi - lo) < tol or abs(xi - hi) < tol)
            for xi, (lo, hi) in zip(x, bounds)]

# -----------------------------
# Calibration routine (sigma_X, sigma_Y, rho)
# -----------------------------
def calibrate_vols(subdf, kappa_X, kappa_Y):
    maturities = subdf["Maturity"].values
    market_vols = subdf["ImpliedVol"].values

    def objective(params):
        sigma_X, sigma_Y, rho = params
        model_vols = gabillon_forward_vol(maturities, kappa_X, kappa_Y, sigma_X, sigma_Y, rho)
        return np.sum((model_vols - market_vols)**2)

    x0 = [0.3, 0.1, 0.0]
    res = minimize(
        objective, x0, bounds=BOUNDS, method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 5000, "maxls": 50},
    )
    return res

def main():
    # -----------------------------
    # Paths
    # -----------------------------
    base_path = os.path.dirname(os.path.abspath(__file__))

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
    # Loop over all dates
    # -----------------------------
    results = []
    skipped = 0
    for d in param_df["Date"]:
        row = param_df[param_df["Date"] == d].iloc[0]
        kappa_X, kappa_Y = row["kappa_X"], row["kappa_Y"]

        subdf = vol_df[vol_df["Date"] == d]
        if subdf.empty:
            skipped += 1
            continue

        res = calibrate_vols(subdf, kappa_X, kappa_Y)
        sigma_X, sigma_Y, rho = res.x

        out = {"Date": d, "sigma_X": sigma_X, "sigma_Y": sigma_Y, "rho": rho,
               "converged": bool(res.success)}
        out.update({f"on_bound_{name}": flag
                    for name, flag in zip(PARAM_NAMES, bound_flags(res.x, BOUNDS))})
        results.append(out)

    # -----------------------------
    # Save results
    # -----------------------------
    vol_param_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, "Gabillon_volatility_calibration_results.csv")
    vol_param_df.to_csv(output_file, index=False)

    # -----------------------------
    # Diagnostics summary
    # -----------------------------
    n = len(vol_param_df)
    print(f"Calibrated {n} dates -> {output_file}")
    if skipped:
        print(f"  skipped {skipped} dates with no matching volatility data")
    print(f"  converged: {vol_param_df['converged'].sum()} / {n} "
          f"({100 * vol_param_df['converged'].mean():.1f}%)")
    print("  parameters resting on a bound:")
    for name in PARAM_NAMES:
        pct = 100 * vol_param_df[f"on_bound_{name}"].mean()
        flag = "   <-- not an estimate, a bound artefact" if pct > 20 else ""
        print(f"    {name:8s} {pct:6.2f}%{flag}")

    print()
    print(vol_param_df.head())


if __name__ == "__main__":
    main()
