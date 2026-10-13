"""Run one full-history EIA curve calibration with current Step1 defaults.

Reads the prepared ACT/365F long table. Saves to a separate directory.
This file does not run data preparation, volatility calibration or simulation.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from Step1_Calibrate_Gabillon_model_parameters import (
    PARAM_NAMES, bound_flags, calibrate_for_date, gabillon_forward,
)

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "Output" / "Step1a_EIA_Preparation" / "EIA_WTI_calibration_input.csv"
OUT = ROOT / "Output" / "Step1c_EIA_Full_Sample"


def main():
    source_hash = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    data = pd.read_csv(INPUT, parse_dates=["Date", "ExpiryDate"])
    assert len(data) == 4276 and data.Date.nunique() == 1069
    assert data.CalibrationEligible.all()
    assert data.groupby("Date").size().eq(4).all()
    assert not data.duplicated(["Date", "Contract"]).any()
    assert np.isfinite(data[["Price", "Maturity"]].to_numpy()).all()
    assert data.Price.gt(0).all() and data.Maturity.ge(0).all()
    np.testing.assert_allclose(data.Maturity, (data.ExpiryDate-data.Date).dt.days/365,
                               atol=1e-15, rtol=1e-14)
    results, price_rows = [], []
    for count, (day, group) in enumerate(data.groupby("Date", sort=True), start=1):
        group = group.sort_values("Maturity")
        assert group.Maturity.is_unique
        res, bounds = calibrate_for_date(group)
        fitted = np.exp(gabillon_forward(group.Maturity.to_numpy(), *res.x))
        errors = fitted-group.Price.to_numpy()
        assert np.isfinite(fitted).all()
        np.testing.assert_allclose(errors @ errors, res.fun, rtol=1e-9, atol=1e-12)
        flags = bound_flags(res.x, bounds)
        lower = [abs(value-bound[0]) < 1e-6 for value,bound in zip(res.x,bounds)]
        upper = [abs(value-bound[1]) < 1e-6 for value,bound in zip(res.x,bounds)]
        record = {"Date": day, **dict(zip(PARAM_NAMES,res.x)),
                  "converged": bool(res.success), "optimizer_message": str(res.message),
                  "iterations": res.nit, "function_evaluations": res.nfev,
                  "RMSE_USD": np.sqrt(np.mean(errors**2)),
                  "max_abs_error_USD": np.max(np.abs(errors)),
                  "any_bound": any(flags),
                  # kappa_Y=0 is an admissible structural case, tracked separately.
                  "other_bound": any(flag for name,flag in zip(PARAM_NAMES,flags)
                                     if name != "kappa_Y") or upper[1]}
        for i, param in enumerate(PARAM_NAMES):
            record.update({f"on_bound_{param}": flags[i],
                           f"on_lower_{param}": lower[i], f"on_upper_{param}": upper[i]})
        results.append(record)
        detail = group[["Date","NearbyRank","Contract","ExpiryDate","Maturity","Price"]].copy()
        detail["FittedPrice"] = fitted
        detail["Error_USD"] = errors
        price_rows.append(detail)
        if count % 100 == 0 or count == 1069:
            print(f"Calibrated {count}/1069 dates", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(results)
    result.to_csv(OUT / "EIA_calibration_results.csv", index=False)
    pd.concat(price_rows).to_csv(OUT / "EIA_fitted_prices.csv", index=False)
    annual = result.groupby(result.Date.dt.year).agg(
        dates=("Date","size"), converged=("converged","sum"),
        mean_RMSE_USD=("RMSE_USD","mean"), median_RMSE_USD=("RMSE_USD","median"),
        max_RMSE_USD=("RMSE_USD","max"),
        any_bound_count=("any_bound","sum"), other_bound_count=("other_bound","sum"))
    annual.to_csv(OUT / "EIA_annual_summary.csv", index_label="Year")
    quantiles = result.RMSE_USD.quantile([0,.5,.9,.95,.99,1])
    bound_summary = pd.DataFrame([
        {"parameter": p,"lower_count": int(result[f"on_lower_{p}"].sum()),
         "upper_count": int(result[f"on_upper_{p}"].sum()),
         "either_pct": 100*result[f"on_bound_{p}"].mean()} for p in PARAM_NAMES])
    bound_summary.to_csv(OUT / "EIA_bound_summary.csv", index=False)
    worst = result.nlargest(10,"RMSE_USD")
    report = [
        "EIA WTI full-sample curve calibration; current Step1 defaults.",
        "2020-01-02 to 2024-04-05; 2020-04-20 excluded because of nonpositive price.",
        "Four nearby contracts/day; ACT/365F to rule-derived CL last-trading dates.",
        "Same fixed initial guess each day; no warm start, multistart or relaxed bounds.",
        f"Input SHA256: {source_hash}",
        f"Dates: {len(result)}; converged: {result.converged.sum()}; failures: {(~result.converged).sum()}.",
        f"Mean RMSE: {result.RMSE_USD.mean():.8f} USD/barrel.",
        f"Any bound: {result.any_bound.sum()} ({100*result.any_bound.mean():.2f}%).",
        f"Bounds excluding kappa_Y=0 alone: {result.other_bound.sum()} ({100*result.other_bound.mean():.2f}%).",
        "kappa_Y=0 is an allowed structural boundary; a bound flag alone does not imply failure.",
        "Failed optimizations are retained and included in the aggregate diagnostics.",
        "All errors are in-sample; four prices for four parameters do not independently validate estimates.",
        "These outputs are a diagnostic, not validated daily economic parameters.",
        "No volatility calibration or simulation was run.", "", "RMSE quantiles:",
        quantiles.to_string(), "", "Bound summary:", bound_summary.to_string(index=False),
        "", "Annual summary:", annual.to_string(), "", "Optimizer stopping reasons:",
        result.optimizer_message.value_counts().to_string(), "", "Worst 10 dates:",
        worst[["Date",*PARAM_NAMES,"RMSE_USD","converged","other_bound"]].to_string(index=False),
    ]
    (OUT / "full_sample_report.txt").write_text("\n".join(report), encoding="utf-8")
    assert hashlib.sha256(INPUT.read_bytes()).hexdigest() == source_hash
    saved = pd.read_csv(OUT / "EIA_calibration_results.csv")
    assert len(saved) == 1069 and saved.Date.is_unique
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
