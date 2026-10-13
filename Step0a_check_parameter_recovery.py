"""Test the existing Step1 calibrator against noise-free Step0 curves.

Run directly; outputs are isolated from market calibration results.
This tests numerical recovery, not risk-neutral simulation consistency.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from Step0_generate_model_data import CASES, TENOR_T, generate
from Step1_Calibrate_Gabillon_model_parameters import (
    PARAM_NAMES, bound_flags, calibrate_for_date, gabillon_forward,
)

OUTPUT = Path(__file__).resolve().parent / "Output" / "Step0a_Parameter_Recovery"
FIT_COLS = ["1M", "3M", "6M", "12M"]
FIT_T = np.array([1 / 12, 3 / 12, 6 / 12, 1.0])
PARAM_ATOL = 1e-3  # absolute error per parameter; includes log-state errors
PRICE_ATOL = 1e-4  # USD per barrel


def run_case(name, parameters):
    futures, _, truth = generate(name, **parameters, write=False)
    rows = []
    for i, curve in futures.iterrows():
        true = truth.iloc[i][PARAM_NAMES].to_numpy(dtype=float)
        market = curve[FIT_COLS].to_numpy(dtype=float)
        subdf = pd.DataFrame({"Maturity": FIT_T, "Price": market})
        res, bounds = calibrate_for_date(subdf)
        lower, upper = np.array(bounds).T
        eligible = bool(np.all((true >= lower) & (true <= upper)))
        predicted = np.exp(gabillon_forward(FIT_T, *res.x))
        # An independent reconstruction catches tenor/state wiring mistakes.
        expected = np.exp(true[2] * np.exp(-true[0] * FIT_T)
                          + true[3] * np.exp(-true[1] * FIT_T))
        np.testing.assert_allclose(market, expected, rtol=1e-12, atol=1e-10)
        heldout = ~np.isin(TENOR_T, FIT_T)
        all_prices = curve[[f"{m}M" for m in range(1, 13)]].to_numpy(dtype=float)
        heldout_prices = np.exp(gabillon_forward(TENOR_T[heldout], *res.x))
        row = {
            "case": name, "Date": curve["Date"],
            "truth_within_bounds": eligible, "converged": bool(res.success),
            "optimizer_message": str(res.message),
            "fit_RMSE_USD": float(np.sqrt(np.mean((market - predicted) ** 2))),
            "heldout_RMSE_USD": float(np.sqrt(np.mean(
                (all_prices[heldout] - heldout_prices) ** 2))),
            "parameters_recovered": bool(np.all(np.abs(res.x - true) <= PARAM_ATOL)),
        }
        for j, (param, flag) in enumerate(zip(PARAM_NAMES, bound_flags(res.x, bounds))):
            row.update({f"true_{param}": true[j], f"estimated_{param}": res.x[j],
                        f"error_{param}": res.x[j] - true[j], f"on_bound_{param}": flag})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    details = pd.concat([run_case(name, params) for name, params in CASES.items()],
                        ignore_index=True)
    details.to_csv(OUTPUT / "parameter_recovery_details.csv", index=False)
    summaries = []
    for name, group in details.groupby("case", sort=False):
        summary = {
            "case": name, "dates": len(group),
            "truth_within_bounds_count": int(group.truth_within_bounds.sum()),
            "converged_count": int(group.converged.sum()),
            "recovered_count": int(group.parameters_recovered.sum()),
            "price_fit_pass_count": int((group.fit_RMSE_USD <= PRICE_ATOL).sum()),
            "mean_fit_RMSE_USD": group.fit_RMSE_USD.mean(),
            "mean_heldout_RMSE_USD": group.heldout_RMSE_USD.mean(),
        }
        for param in PARAM_NAMES:
            summary[f"MAE_{param}"] = group[f"error_{param}"].abs().mean()
            summary[f"on_bound_count_{param}"] = int(group[f"on_bound_{param}"].sum())
        summaries.append(summary)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTPUT / "parameter_recovery_summary.csv", index=False)
    report = [
        "Noise-free Step0 -> current Step1 calibration algorithm",
        f"Parameter recovery tolerance: absolute error <= {PARAM_ATOL} for every parameter.",
        f"Price fit tolerance: RMSE <= {PRICE_ATOL} USD/barrel.",
        "Fit tenors: 1M/3M/6M/12M. Held-out tenors: 2M/4M/5M/7M/8M/9M/10M/11M.",
        "Both cases have reachable kappas; per-date state bounds are checked separately.",
        "Case B tests kappa_Y=0 at Step1's lower bound; exact recovery is now permitted.",
        "Failure can reflect optimizer stopping, poor conditioning, bounds or ambiguity;",
        "this baseline alone does not establish structural non-identifiability.",
        "No noise sensitivity or volatility recovery experiment is performed here.",
        "",
        summary_df.to_string(index=False),
    ]
    (OUTPUT / "recovery_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
