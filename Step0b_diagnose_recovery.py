"""Diagnose Case A without changing the production calibrator.

Compare solver stopping, analytic derivatives, tenor count and starting points.
All experiments use the same price residuals and original parameter bounds.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from Step0_generate_model_data import CASES, TENOR_T, generate
from Step1_Calibrate_Gabillon_model_parameters import PARAM_NAMES, gabillon_forward

OUT = Path(__file__).resolve().parent / "Output" / "Step0b_Recovery_Diagnosis"
FIT_IDX = np.array([0, 2, 5, 11])


def residual(p, t, prices):
    return np.exp(gabillon_forward(t, *p)) - prices


def jacobian(p, t, prices):
    kx, ky, x, y = p
    bx, by = np.exp(-kx * t), np.exp(-ky * t)
    f = np.exp(x * bx + y * by)
    return f[:, None] * np.column_stack((-t * x * bx, -t * y * by, bx, by))


def solve(method, x0, bounds, t, prices):
    if method == "lbfgsb_tight_analytic":
        def fun(p):
            r = residual(p, t, prices)
            return r @ r, 2 * jacobian(p, t, prices).T @ r
        return minimize(fun, x0, jac=True, bounds=bounds, method="L-BFGS-B",
                        options={"ftol": 1e-15, "gtol": 1e-12,
                                 "maxiter": 5000, "maxls": 50})
    return least_squares(residual, x0, jac=jacobian, args=(t, prices),
                         bounds=np.array(bounds).T, x_scale="jac",
                         ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=3000)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f, _, truth = generate("caseA_reachable", **CASES["caseA_reachable"], write=False)
    baseline = pd.read_csv(OUT.parent / "Step0a_Parameter_Recovery" /
                           "parameter_recovery_details.csv")
    baseline = baseline[baseline.case == "caseA_reachable"].reset_index(drop=True)
    assert list(baseline.Date) == f.Date.dt.strftime("%Y-%m-%d").tolist()
    experiments, conditioning = [], []
    for i, row in f.iterrows():
        prices = row[[f"{m}M" for m in range(1, 13)]].to_numpy(float)
        true = truth.iloc[i][PARAM_NAMES].to_numpy(float)
        level = np.log(prices[0])
        bounds = [(0.01, 5), (0.01, 1), (-2, 2), (level - 1, level + 1)]
        x0 = np.array([1, .2, 0, level])
        # Compare analytic derivatives to central differences at the true state.
        eps = 1e-6
        numeric = np.column_stack([
            (residual(true + np.eye(4)[j] * eps, TENOR_T, prices)
             - residual(true - np.eye(4)[j] * eps, TENOR_T, prices)) / (2 * eps)
            for j in range(4)])
        np.testing.assert_allclose(jacobian(true, TENOR_T, prices), numeric,
                                   rtol=1e-5, atol=1e-7)
        for label, idx in [("4", FIT_IDX), ("12", np.arange(12))]:
            singular = np.linalg.svd(jacobian(true, TENOR_T[idx], prices[idx]),
                                     compute_uv=False)
            conditioning.append({"Date": row.Date, "tenors": label,
                                 "X0": true[2], "smallest_singular_value": singular[-1],
                                 "condition_number": singular[0] / singular[-1]})
        variants = [("lbfgsb_tight_analytic", FIT_IDX),
                    ("least_squares_4", FIT_IDX),
                    ("least_squares_12", np.arange(12))]
        # Deterministic alternative starts on a subset plus the worst baseline day.
        if i % 21 == 0 or i == baseline.fit_RMSE_USD.idxmax():
            variants += [("multistart_4", FIT_IDX)]
        for method, idx in variants:
            starts = [x0]
            if method == "multistart_4":
                starts += [np.array([kx, ky, x, level - x])
                           for kx, ky, x in [(2.5, .1, -.2), (.5, .03, .2), (4, .4, -.5)]]
                starts = [np.clip(p, *np.array(bounds).T) for p in starts]
            for start_id, start in enumerate(starts):
                res = solve(method, start, bounds, TENOR_T[idx], prices[idx])
                err = res.x - true
                out = {"Date": row.Date, "method": method, "start": start_id,
                       "converged": bool(res.success), "message": str(res.message),
                       "nfev": res.nfev,
                       "fit_RMSE_USD": np.sqrt(np.mean(residual(res.x, TENOR_T[idx], prices[idx])**2)),
                       "all12_RMSE_USD": np.sqrt(np.mean(residual(res.x, TENOR_T, prices)**2)),
                       "max_parameter_error": np.max(np.abs(err)),
                       "recovered": bool(np.all(np.abs(err) <= .001))}
                for j, name in enumerate(PARAM_NAMES):
                    out.update({f"true_{name}": true[j], f"estimated_{name}": res.x[j],
                                f"error_{name}": err[j]})
                experiments.append(out)
        if i % 42 == 0:
            print(f"Completed {i + 1}/252 dates", flush=True)
    detail = pd.DataFrame(experiments)
    detail.to_csv(OUT / "solver_comparison_details.csv", index=False)
    cond = pd.DataFrame(conditioning)
    cond.to_csv(OUT / "jacobian_conditioning.csv", index=False)
    # Select by residual only, never by distance to known truth. Retry every
    # date that fails the recovery threshold in this diagnostic experiment.
    robust = []
    for i, curve in f.iterrows():
        date = curve.Date
        prices = curve[[f"{m}M" for m in range(1, 13)]].to_numpy(float)
        true = truth.iloc[i][PARAM_NAMES].to_numpy(float)
        level = np.log(prices[0])
        bounds = [(0.01, 5), (0.01, 1), (-2, 2), (level - 1, level + 1)]
        candidates = detail[(detail.Date == date) & (detail.method != "least_squares_12")]
        best = candidates.loc[candidates.fit_RMSE_USD.idxmin()]
        start = best[[f"estimated_{p}" for p in PARAM_NAMES]].to_numpy(float)
        fitted = solve("least_squares_4", start, bounds, TENOR_T[FIT_IDX], prices[FIT_IDX])
        if np.max(np.abs(fitted.x - true)) > .001:
            for kx, ky, x in [(2.5, .1, -.2), (.5, .03, .2), (4, .4, -.5), (1.8, .07, .3)]:
                alt = np.clip([kx, ky, x, level - x], *np.array(bounds).T)
                trial = solve("least_squares_4", alt, bounds, TENOR_T[FIT_IDX], prices[FIT_IDX])
                if np.sum(trial.fun**2) < np.sum(fitted.fun**2):
                    fitted = trial
        robust.append({"Date": date, "recovered": np.max(np.abs(fitted.x - true)) <= .001,
                       "max_parameter_error": np.max(np.abs(fitted.x - true)),
                       "fit_RMSE_USD": np.sqrt(np.mean(fitted.fun**2)),
                       **{f"estimated_{p}": fitted.x[j] for j, p in enumerate(PARAM_NAMES)}})
    robust_df = pd.DataFrame(robust)
    robust_df.to_csv(OUT / "polished_recovery.csv", index=False)
    # Small reproducible noise experiment on weak/typical/strong short-factor
    # dates. Local starts at the noise-free fitted solution isolate sensitivity
    # from cold-start failures; this is not an operational calibration benchmark.
    magnitudes = truth.X0.abs().to_numpy()
    sample_indices = {"weak": int(np.argmin(magnitudes)),
                      "typical": int(np.argsort(magnitudes)[len(magnitudes)//2]),
                      "strong": int(np.argmax(magnitudes))}
    rng = np.random.default_rng(42)
    noise_rows = []
    for label, i in sample_indices.items():
        true = truth.iloc[i][PARAM_NAMES].to_numpy(float)
        prices = f.iloc[i][[f"{m}M" for m in range(1, 13)]].to_numpy(float)
        start = robust_df.iloc[i][[f"estimated_{p}" for p in PARAM_NAMES]].to_numpy(float)
        for replicate in range(10):
            noisy = prices + rng.normal(0, .005, 12)
            level = np.log(noisy[0])
            bounds = [(0.01, 5), (0.01, 1), (-2, 2), (level-1, level+1)]
            for label_t, idx in [("4", FIT_IDX), ("12", np.arange(12))]:
                fit = solve("least_squares_4", np.clip(start, *np.array(bounds).T),
                            bounds, TENOR_T[idx], noisy[idx])
                noise_rows.append({"state": label, "Date": f.iloc[i].Date,
                                   "tenors": label_t, "replicate": replicate,
                                   "X0": true[2], "price_noise_SD_USD": .005,
                                   "abs_error_kappa_X": abs(fit.x[0]-true[0]),
                                   "abs_error_kappa_Y": abs(fit.x[1]-true[1]),
                                   "fit_RMSE_USD": np.sqrt(np.mean(fit.fun**2))})
    noise_df = pd.DataFrame(noise_rows)
    noise_df.to_csv(OUT / "small_noise_sensitivity.csv", index=False)
    noise_summary = noise_df.groupby(["state", "tenors"]).agg(
        mean_abs_error_kappa_X=("abs_error_kappa_X", "mean"),
        max_abs_error_kappa_X=("abs_error_kappa_X", "max"),
        mean_RMSE=("fit_RMSE_USD", "mean"))
    summary = detail.groupby("method").agg(
        runs=("Date", "size"), recovered=("recovered", "sum"),
        converged=("converged", "sum"), mean_RMSE=("fit_RMSE_USD", "mean"),
        max_RMSE=("fit_RMSE_USD", "max"),
        mean_max_parameter_error=("max_parameter_error", "mean"),
        worst_parameter_error=("max_parameter_error", "max"))
    summary.to_csv(OUT / "solver_comparison_summary.csv")
    conditioning_summary = cond.groupby("tenors").condition_number.agg(["min", "median", "max"])
    report = ["Case A: same data, price objective and bounds; solver variants only.",
              "No truth parameters are used as solver starting points.",
              "Recovery: all four absolute parameter errors <= 0.001.",
              "Baseline stopping reasons:", baseline.optimizer_message.value_counts().to_string(),
              "", "Solver comparison:", summary.to_string(), "",
              "Jacobian condition number at truth (raw parameter units):",
              conditioning_summary.to_string(), "",
              f"Residual-selected solutions with polishing/retries: {robust_df.recovered.sum()}/252 recovered.",
              f"Worst polished parameter error: {robust_df.max_parameter_error.max():.6g}",
              "Retries are triggered by known-truth errors for diagnosis only, not production logic.",
              "", "Price noise SD = $0.005, 10 draws/state, local starts at noise-free estimates:",
              noise_summary.to_string(), "",
              "Three selected states are illustrative, not a population-wide noise study.",
              "Full rank locally does not prove global uniqueness. Large condition numbers",
              "indicate sensitivity in these units, not a measured noise experiment.",
              "The initial X0=0 makes the initial kappa_X derivative exactly zero.",
              "Case B bounds and risk-neutral consistency are outside this diagnostic."]
    (OUT / "diagnosis_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
