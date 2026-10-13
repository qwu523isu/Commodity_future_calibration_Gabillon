"""Compare kappa_Y=0 against saved four-parameter EIA calibration.

Run this file only. Reuses Step1 with a fixed (0,0) kappa_Y bound; all other
bounds, initial values, objective and optimizer settings are unchanged.
"""
from pathlib import Path
import hashlib
import html
import numpy as np
import pandas as pd
from Step1_Calibrate_Gabillon_model_parameters import (
    PARAM_NAMES, bound_flags, calibrate_for_date, gabillon_forward,
)
from Step2a_review_EIA_results import chart, table

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "Output" / "Step1a_EIA_Preparation" / "EIA_WTI_calibration_input.csv"
BASELINE = ROOT / "Output" / "Step1c_EIA_Full_Sample" / "EIA_calibration_results.csv"
OUT = ROOT / "Output" / "Step1d_EIA_Three_Parameter_Comparison"
FREE_PARAMS = ["kappa_X", "X0", "Y0"]


def main():
    hashes = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in [INPUT,BASELINE]}
    data = pd.read_csv(INPUT,parse_dates=["Date"])
    baseline = pd.read_csv(BASELINE,parse_dates=["Date"]).sort_values("Date")
    assert len(data)==4276 and data.Date.nunique()==1069
    rows, prices = [], []
    for count,(day,group) in enumerate(data.groupby("Date",sort=True),start=1):
        group = group.sort_values("Maturity")
        assert len(group)==4 and group.Price.gt(0).all()
        log_level = np.log(group.Price.iloc[0])
        bounds = [(0.01,5),(0,0),(-2,2),(log_level-1,log_level+1)]
        result, used_bounds = calibrate_for_date(group,bounds=bounds)
        assert result.x[1]==0
        fitted = np.exp(gabillon_forward(group.Maturity.to_numpy(),*result.x))
        error = fitted-group.Price.to_numpy()
        np.testing.assert_allclose(error @ error,result.fun,rtol=1e-9,atol=1e-12)
        flags = bound_flags(result.x,used_bounds)
        free_flags = [flags[j] for j in [0,2,3]]
        row = {"Date":day,**dict(zip(PARAM_NAMES,result.x)),
               "RMSE_USD":np.sqrt(np.mean(error**2)),
               "converged":bool(result.success),"optimizer_message":str(result.message),
               "iterations":result.nit,"other_bound":any(free_flags)}
        row.update({f"on_bound_{p}":flags[j] for j,p in enumerate(PARAM_NAMES) if p!="kappa_Y"})
        rows.append(row)
        price = group[["Date","Contract","Maturity","Price"]].copy()
        price["FittedPrice"] = fitted
        price["Error_USD"] = error
        prices.append(price)
        if count%200==0 or count==1069:
            print(f"Three-parameter fits: {count}/1069",flush=True)
    three = pd.DataFrame(rows)
    assert three.Date.tolist()==baseline.Date.tolist()
    OUT.mkdir(parents=True,exist_ok=True)
    three.to_csv(OUT / "three_parameter_results.csv",index=False)
    pd.concat(prices).to_csv(OUT / "three_parameter_fitted_prices.csv",index=False)
    cols = ["Date",*PARAM_NAMES,"RMSE_USD","converged","other_bound"]
    paired = baseline[cols].merge(three[cols],on="Date",suffixes=("_four","_three"),validate="one_to_one")
    paired["RMSE_change_USD"] = paired.RMSE_USD_three-paired.RMSE_USD_four
    paired.to_csv(OUT / "daily_model_comparison.csv",index=False)
    summary, stability, annual = [], [], []
    for name,frame in [("four_parameter",baseline),("three_parameter",three)]:
        summary.append({"model":name,"dates":len(frame),"converged":int(frame.converged.sum()),
                        "mean_RMSE_USD":frame.RMSE_USD.mean(),
                        "median_RMSE_USD":frame.RMSE_USD.median(),
                        "p95_RMSE_USD":frame.RMSE_USD.quantile(.95),
                        "max_RMSE_USD":frame.RMSE_USD.max(),
                        "free_parameter_bound_pct":100*frame.other_bound.mean()})
        both_converged = frame.converged & frame.converged.shift(1,fill_value=False)
        for p in FREE_PARAMS:
            change = frame[p].diff().abs()
            stability.append({"model":name,"parameter":p,
                              "mean_abs_change":change.mean(),
                              "median_abs_change":change.median(),
                              "p95_abs_change":change.quantile(.95),
                              "max_abs_change":change.max(),
                              "mean_abs_change_converged_pairs":change[both_converged].mean()})
        yearly = frame.groupby(frame.Date.dt.year).agg(
            dates=("Date","size"),converged=("converged","sum"),
            mean_RMSE_USD=("RMSE_USD","mean"),
            bound_pct=("other_bound",lambda v:100*v.mean())).reset_index()
        yearly["model"] = name
        annual.append(yearly)
    summary = pd.DataFrame(summary)
    stability = pd.DataFrame(stability)
    annual = pd.concat(annual,ignore_index=True)
    summary.to_csv(OUT / "model_summary.csv",index=False)
    stability.to_csv(OUT / "parameter_stability.csv",index=False)
    annual.to_csv(OUT / "annual_model_comparison.csv",index=False)
    gain = int((paired.RMSE_change_USD < -1e-6).sum())
    loss = int((paired.RMSE_change_USD > 1e-6).sum())
    report = [
        "Three-parameter versus saved four-parameter EIA model.",
        "Only change: kappa_Y bound fixed to (0,0). Fixed parameter excluded from bound percentages.",
        "Both models use the same remaining bounds, initial guesses and optimization settings.",
        "No multistart, warm start, volatility calibration or simulation.",
        "All dates, including optimizer failures, are included in aggregate diagnostics.",
        "Stability uses absolute changes between adjacent observations, not calendar-day rates.",
        "kappa_Y=0 alone is also excluded from the four-parameter free-bound comparison.",
        f"Three-parameter lower/higher/equivalent RMSE (tolerance $1e-6): {gain}/{loss}/{1069-gain-loss} dates.",
        "The three-parameter family is nested inside the four-parameter family.",
        "Better three-parameter fit on some dates reveals four-parameter solver limitations,",
        "not proof that the restricted model has a better global minimum.",
        "Lower parameter changes can result from boundary saturation; inspect bounds alongside stability.",
        "This is an in-sample comparison, not predictive or out-of-sample validation.",
        "Source EIA contract maturities remain rule-derived.",
        f"Prepared input SHA256: {hashes[INPUT]}",f"Baseline SHA256: {hashes[BASELINE]}",
        "", "Model summary:",summary.to_string(index=False),
        "", "Parameter stability:",stability.to_string(index=False),
        "", "Annual comparison:",annual.to_string(index=False),
        "", "Largest increases in RMSE:",paired.nlargest(10,"RMSE_change_USD")[
            ["Date","RMSE_USD_four","RMSE_USD_three","RMSE_change_USD"]].to_string(index=False),
    ]
    (OUT / "three_parameter_comparison_report.txt").write_text('\n'.join(report),encoding="utf-8")
    chunks = ['<!doctype html><html><head><meta charset="utf-8"><title>EIA model comparison</title><style>body{font:16px Arial;max-width:1200px;margin:28px auto;color:#1f2937}svg{width:100%;max-width:900px}table{border-collapse:collapse;font-size:13px}th,td{padding:7px;border:1px solid #ddd}pre{white-space:pre-wrap}</style></head><body><h1>EIA: three versus four parameters</h1><p>Only kappa_Y fixed to zero. Other settings unchanged. In-sample comparison.</p>',
              '<h2>Model summary</h2>',table(summary),'<h2>Parameter stability</h2>',table(stability)]
    ordinal = three.Date.map(pd.Timestamp.toordinal).to_numpy()
    for p in FREE_PARAMS+["RMSE_USD"]:
        chunks += [f'<h2>{p}</h2>',chart([
            ('Four parameters',ordinal,baseline[p].to_numpy(),'#2563eb'),
            ('Three parameters',ordinal,three[p].to_numpy(),'#d97706')],'Date',p)]
    chunks += ['<h2>Annual comparison</h2>',table(annual),'<h2>Notes and details</h2><pre>',
               html.escape('\n'.join(report)),'</pre></body></html>']
    (OUT / "model_comparison.html").write_text(''.join(chunks),encoding="utf-8")
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==v for p,v in hashes.items())
    saved = pd.read_csv(OUT / "three_parameter_results.csv")
    assert len(saved)==1069 and saved.kappa_Y.eq(0).all()
    print('\n'.join(report),flush=True)


if __name__ == "__main__":
    main()
