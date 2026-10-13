import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize

# -----------------------------
# Gabillon forward log-price function
# -----------------------------
def gabillon_forward(T, kappa_X, kappa_Y, X0, Y0):
    BX = np.exp(-kappa_X * T)
    BY = np.exp(-kappa_Y * T)
    return X0 * BX + Y0 * BY

# -----------------------------
# Bound diagnostics
# -----------------------------
PARAM_NAMES = ["kappa_X", "kappa_Y", "X0", "Y0"]

def bound_flags(x, bounds, tol=1e-6):
    """One bool per parameter: is it resting on either of its bounds?"""
    return [bool(abs(xi - lo) < tol or abs(xi - hi) < tol)
            for xi, (lo, hi) in zip(x, bounds)]

# -----------------------------
# Calibration routine for one date
# -----------------------------
def calibrate_for_date(subdf, bounds=None):
    T = subdf["Maturity"].values
    F_mkt = subdf["Price"].values
    logS0 = np.log(F_mkt[0])

    def objective(params):
        kappa_X, kappa_Y, X0, Y0 = params
        BX = np.exp(-kappa_X * T)
        BY = np.exp(-kappa_Y * T)
        fitted = np.exp(X0 * BX + Y0 * BY)
        residual = fitted - F_mkt
        jacobian = fitted[:, None] * np.column_stack(
            (-T * X0 * BX, -T * Y0 * BY, BX, BY)
        )
        return residual @ residual, 2 * jacobian.T @ residual

    x0 = [1.0, 0.2, 0.0, logS0]
    if bounds is None:
        bounds = [(0.01, 5), (0.0, 1), (-2, 2), (logS0 - 1, logS0 + 1)]
    res = minimize(
        objective, x0, jac=True, bounds=bounds, method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 5000, "maxls": 50},
    )
    return res, bounds

def main():
    # -----------------------------
    # Paths
    # -----------------------------
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_path, "Input_data", "wti_futures_bbg_historical_constant_maturity.csv")
    output_dir = os.path.join(base_path, "Output", "Step1_Calibrate_Gabillon_model_parameters")
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(input_file, parse_dates=["Date"])

    # Select subset maturities (1M, 3M, 6M, 12M)
    subset_cols = ["1M", "3M", "6M", "12M"]
    maturity_map = {"1M": 1/12, "3M": 3/12, "6M": 6/12, "12M": 12/12}

    # Reshape into long format, mapping maturity labels to year fractions
    long_df = df.melt(id_vars="Date",
                      value_vars=subset_cols,
                      var_name="Maturity",
                      value_name="Price")
    long_df["Maturity"] = long_df["Maturity"].map(maturity_map)

    print(f"Loaded {len(df)} dates from {os.path.basename(input_file)}")
    print(f"Fitting maturities: {subset_cols}")

    # -----------------------------
    # Loop over all dates and store calibration results
    # -----------------------------
    results = []
    for d, subdf in long_df.groupby("Date"):
        res, bounds = calibrate_for_date(subdf)
        kappa_X, kappa_Y, X0, Y0 = res.x

        row = {"Date": d, "kappa_X": kappa_X, "kappa_Y": kappa_Y, "X0": X0, "Y0": Y0,
               "converged": bool(res.success)}
        row.update({f"on_bound_{name}": flag
                    for name, flag in zip(PARAM_NAMES, bound_flags(res.x, bounds))})
        results.append(row)

    param_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, "Gabillon_calibration_results.csv")
    param_df.to_csv(output_file, index=False)

    # -----------------------------
    # Diagnostics summary
    # -----------------------------
    n = len(param_df)
    print(f"\nCalibrated {n} dates -> {output_file}")
    print(f"  converged: {param_df['converged'].sum()} / {n} "
          f"({100 * param_df['converged'].mean():.1f}%)")
    print("  parameters resting on a bound:")
    for name in PARAM_NAMES:
        pct = 100 * param_df[f"on_bound_{name}"].mean()
        flag = "   <-- not an estimate, a bound artefact" if pct > 20 else ""
        print(f"    {name:8s} {pct:6.2f}%{flag}")

    print()
    print(param_df.head())


if __name__ == "__main__":
    main()
