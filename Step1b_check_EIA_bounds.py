"""Small boundary-sensitivity experiment on the three EIA sample dates.

Defaults, objective, optimizer and cold start are unchanged. Wider bounds
are experimental, not new production defaults. No negative kappas allowed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from Step1_Calibrate_Gabillon_model_parameters import (
    PARAM_NAMES, bound_flags, calibrate_for_date, gabillon_forward,
)
from Step1a_prepare_EIA_data import CASE_DATES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Output" / "Step1b_EIA_Bound_Sensitivity"


def main():
    data = pd.read_csv(ROOT / "Output" / "Step1a_EIA_Preparation" /
                       "EIA_WTI_calibration_input.csv", parse_dates=["Date"])
    rows, prices = [], []
    for day in CASE_DATES:
        group = data[data.Date == pd.Timestamp(day)].sort_values("Maturity")
        assert len(group) == 4
        level = np.log(group.Price.iloc[0])
        variants = [
            ("baseline", 5, 2, 1),
            ("Y_width_2", 5, 2, 2),
            ("Y_width_5", 5, 2, 5),
            ("X_width_6_Y_width_5", 5, 6, 5),
            ("kappa_X_max_10", 10, 2, 1),
            ("kappa_X_max_20", 20, 2, 1),
            ("kappa_X_max_50", 50, 2, 1),
        ]
        for name, max_kx, width_x, width_y in variants:
            bounds = [(0.01, max_kx), (0, 1), (-width_x, width_x),
                      (level-width_y, level+width_y)]
            result, used_bounds = calibrate_for_date(group, bounds=bounds)
            estimate = np.exp(gabillon_forward(group.Maturity.to_numpy(), *result.x))
            assert np.isfinite(estimate).all()
            flags = bound_flags(result.x, used_bounds)
            row = {"Date": day, "variant": name, "kappa_X_upper": max_kx,
                   "X0_half_width": width_x, "Y0_half_width": width_y,
                   **dict(zip(PARAM_NAMES, result.x)),
                   "RMSE_USD": np.sqrt(np.mean((estimate-group.Price.to_numpy())**2)),
                   "converged": bool(result.success), "optimizer_message": str(result.message),
                   **{f"on_bound_{p}": flag for p, flag in zip(PARAM_NAMES, flags)}}
            # Extrapolation illustrates sensitivity only; these are not validated prices.
            row["extrapolated_1y_price"] = float(np.exp(gabillon_forward(1, *result.x)))
            rows.append(row)
            detail = group[["Date", "Contract", "Maturity", "Price"]].copy()
            detail["variant"] = name
            detail["FittedPrice"] = estimate
            detail["Error_USD"] = estimate-group.Price.to_numpy()
            prices.append(detail)
    OUT.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    baseline = results[results.variant == "baseline"].set_index("Date")
    results["baseline_RMSE_USD"] = results.Date.map(baseline.RMSE_USD)
    results["RMSE_improvement_pct"] = 100*(1-results.RMSE_USD/results.baseline_RMSE_USD)
    results.to_csv(OUT / "bound_sensitivity_results.csv", index=False)
    pd.concat(prices).to_csv(OUT / "bound_sensitivity_prices.csv", index=False)
    original = pd.read_csv(ROOT / "Output" / "Step1a_EIA_Preparation" /
                           "sample_calibration_results.csv").set_index("Date")
    np.testing.assert_allclose(baseline.loc[CASE_DATES, PARAM_NAMES].to_numpy(float),
                               original.loc[CASE_DATES, PARAM_NAMES].to_numpy(float),
                               rtol=1e-8, atol=1e-8)
    display = results[["Date", "variant", *PARAM_NAMES, "RMSE_USD", "converged",
                       "on_bound_kappa_X", "on_bound_kappa_Y", "on_bound_X0",
                       "on_bound_Y0", "extrapolated_1y_price"]]
    report = ["EIA sample-date boundary sensitivity; baseline reproduced within CSV round-trip tolerance.",
              "Same cold start and price objective in every run. No multistart or warm start.",
              "Default production bounds are unchanged. All kappas remain nonnegative.",
              "kappa_Y=0 is an allowed structural case, not automatically an error.",
              "These are local optimizer results, not certified global minima.",
              "One-year prices are extrapolations beyond observed tenors, for sensitivity only.",
              "No full-history calibration is performed by this script.", "",
              "Findings:",
              "2020: wider state bounds shift the active constraint and give only a small fit improvement.",
              "2022: kappa_X stays at each tested upper bound through 50; no stable interior estimate.",
              "2024: a wider speed bound permits an almost exact fit with kappa_X about 22.41.",
              "Four parameters fitting four prices exactly is not independent validation.",
              "A nested wider bound can yield a worse cold-start result (2024 Y_width_2),",
              "which reveals remaining optimizer sensitivity rather than a worse feasible minimum.",
              "Recommendation: keep default bounds; do not widen them solely to remove flags.",
              "Full-history output, if produced next, should be a diagnostic with bound flags,",
              "not validated daily economic parameter estimates.", "",
              display.to_string(index=False)]
    (OUT / "bound_sensitivity_report.txt").write_text("\n".join(report),encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
