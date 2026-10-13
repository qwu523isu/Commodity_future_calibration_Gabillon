"""Read-only review of saved EIA full-history fits; no optimization is run."""
from pathlib import Path
import hashlib
import html
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Output" / "Step1c_EIA_Full_Sample"
OUT = ROOT / "Output" / "Step2a_EIA_Result_Review"
PARAMS = ["kappa_X", "kappa_Y", "X0", "Y0"]


def chart(series, x_label, y_label, marks=None):
    """Standalone SVG chart with actual observation coordinates."""
    xs = np.concatenate([np.asarray(s[1],float) for s in series])
    ys = np.concatenate([np.asarray(s[2],float) for s in series])
    xmin,xmax = xs.min(), xs.max()
    ymin,ymax = ys.min(), ys.max()
    xspan = max(xmax-xmin, 1e-12)
    pad = max((ymax-ymin)*.08, 1e-5)
    ymin,ymax = ymin-pad,ymax+pad
    x = lambda v: 75+(v-xmin)/xspan*780
    y = lambda v: 275-(v-ymin)/(ymax-ymin)*220
    svg = ['<svg viewBox="0 0 900 340" role="img" aria-label="'+html.escape(y_label)+'">',
           '<rect width="900" height="340" fill="white"/>',
           '<path d="M75 55 V275 H855" fill="none" stroke="#666"/>',
           '<text x="75" y="22" font-size="14">'+html.escape(y_label)+'</text>']
    for v in np.linspace(ymin,ymax,5):
        svg += [f'<path d="M75 {y(v):.2f} H855" stroke="#eee"/>',
                f'<text x="65" y="{y(v)+4:.2f}" text-anchor="end" font-size="12">{v:.4g}</text>']
    for j,(name,sx,sy,color) in enumerate(series):
        points = ' '.join(f'{x(a):.2f},{y(b):.2f}' for a,b in zip(sx,sy))
        svg += [f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"/>',
                f'<text x="{75+j*250}" y="42" fill="{color}" font-size="13">{html.escape(name)}</text>']
        if len(sx)<=8:
            svg += [f'<circle cx="{x(a):.2f}" cy="{y(b):.2f}" r="4" fill="{color}"/>' for a,b in zip(sx,sy)]
    if marks is not None:
        svg += [f'<circle cx="{x(a):.2f}" cy="{y(b):.2f}" r="3" fill="#dc2626"/>' for a,b in marks]
    if x_label == "Date":
        for dt in pd.date_range("2020-01-01","2024-01-01",freq="YS"):
            ordinal = dt.toordinal()
            if xmin<=ordinal<=xmax:
                svg.append(f'<text x="{x(ordinal):.2f}" y="298" text-anchor="middle" font-size="12">{dt.year}</text>')
    else:
        for v in sorted(set(xs)):
            svg.append(f'<text x="{x(v):.2f}" y="298" text-anchor="middle" font-size="12">{v:.0f}</text>')
    svg += ['<text x="450" y="326" text-anchor="middle" font-size="14">'+html.escape(x_label)+'</text>','</svg>']
    return ''.join(svg)


def table(frame):
    return frame.to_html(index=False,float_format=lambda v:f"{v:.6f}",escape=True)


def main():
    paths = [SOURCE / "EIA_calibration_results.csv", SOURCE / "EIA_fitted_prices.csv"]
    hashes = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    results = pd.read_csv(paths[0],parse_dates=["Date"]).sort_values("Date")
    prices = pd.read_csv(paths[1],parse_dates=["Date","ExpiryDate"])
    assert len(results)==1069 and results.Date.is_unique and len(prices)==4276
    np.testing.assert_allclose(prices.FittedPrice-prices.Price,prices.Error_USD,atol=1e-12)
    computed = prices.groupby("Date").Error_USD.apply(lambda e:np.sqrt(np.mean(e**2)))
    np.testing.assert_allclose(results.set_index("Date").RMSE_USD,computed,atol=1e-12)
    joined = prices.merge(results[["Date",*PARAMS]],on="Date",validate="many_to_one")
    reconstructed = np.exp(joined.X0*np.exp(-joined.kappa_X*joined.Maturity)
                           +joined.Y0*np.exp(-joined.kappa_Y*joined.Maturity))
    np.testing.assert_allclose(reconstructed,joined.FittedPrice,rtol=1e-12,atol=1e-11)
    failures = results[~results.converged].sort_values("RMSE_USD",ascending=False)
    worst = results.nlargest(10,"RMSE_USD")
    monthly = results.groupby(results.Date.dt.to_period("M")).agg(
        dates=("Date","size"),failed=("converged",lambda v:int((~v).sum())),
        mean_RMSE=("RMSE_USD","mean"),max_RMSE=("RMSE_USD","max"),
        other_bound_pct=("other_bound",lambda v:100*v.mean()),
        kappa_X_upper_pct=("on_upper_kappa_X",lambda v:100*v.mean()),
        Y0_bound_pct=("on_bound_Y0",lambda v:100*v.mean()))
    changes = results[["Date",*PARAMS]].copy()
    for p in PARAMS:
        changes[f"delta_{p}"] = changes[p].diff()
        changes[f"abs_delta_{p}"] = changes[p].diff().abs()
    changes = changes.iloc[1:]
    jumps = pd.concat([changes.nlargest(5,f"abs_delta_{p}").assign(parameter=p)
                       for p in PARAMS],ignore_index=True)
    by_year = results.groupby(results.Date.dt.year).agg(
        dates=("Date","size"), failed=("converged",lambda v:int((~v).sum())),
        mean_RMSE=("RMSE_USD","mean"), other_bound_pct=("other_bound",lambda v:100*v.mean()))
    expiry_day = prices[prices.NearbyRank==1].set_index("Date").Maturity.eq(0)
    failed_summary = failures.groupby(failures.Date.dt.year).size()
    columns = ["Date",*PARAMS,"RMSE_USD","max_abs_error_USD","converged",
               "other_bound","optimizer_message"]
    OUT.mkdir(parents=True,exist_ok=True)
    failures.to_csv(OUT / "nonconverged_dates.csv",index=False)
    worst.to_csv(OUT / "worst_fit_dates.csv",index=False)
    monthly.to_csv(OUT / "monthly_diagnostics.csv",index_label="Month")
    changes.to_csv(OUT / "daily_parameter_changes.csv",index=False)
    report = [
        "Read-only review of EIA saved calibration results; no recalibration.",
        "Saved fitted prices and RMSE independently reconciled to saved parameters.",
        f"Dates: {len(results)}; nonconverged: {len(failures)}.",
        "Nonconverged dates by year:",failed_summary.to_string(),
        f"Nonconverged RMSE median/max: {failures.RMSE_USD.median():.6f} / {failures.RMSE_USD.max():.6f} USD.",
        f"Converged RMSE median/max: {results.loc[results.converged,'RMSE_USD'].median():.6f} / {results.loc[results.converged,'RMSE_USD'].max():.6f} USD.",
        f"Other-bound dates: {results.other_bound.sum()} ({100*results.other_bound.mean():.2f}%).",
        f"Top 10 error dates in 2020: {(worst.Date.dt.year==2020).sum()}/10.",
        f"Top 10 error dates marked converged: {worst.converged.sum()}/10.",
        f"Expiry-day observations: {expiry_day.sum()}; among top 10 errors: {worst.Date.map(expiry_day).sum()}.",
        "Day changes mean adjacent observed trading dates, not constant calendar-day intervals.",
        "Bound flags and solver success do not establish economic parameter reliability.",
        "kappa_Y=0 alone is treated as an admissible structural boundary.",
        "Expiry dates are rule-derived; these diagnostics do not independently verify every contract identity.",
        "No causal attribution of fit problems to a particular event is established here.",
        "Recommendation: do not proceed to market volatility/simulation using these parameters as validated.",
        "A separate three-parameter kappa_Y=0 comparison is the next simple model experiment,",
        "with the current four-parameter results preserved as the baseline.", "",
        "Annual summary:",by_year.to_string(),"","All nonconverged dates:",
        failures[columns].to_string(index=False),"","Worst 10 dates:",
        worst[columns].to_string(index=False),"","Largest parameter changes (5 per parameter):",
        jumps[["Date","parameter",*PARAMS,*[f"delta_{p}" for p in PARAMS]]].to_string(index=False),
    ]
    (OUT / "result_review_report.txt").write_text('\n'.join(report),encoding="utf-8")
    ordinal = results.Date.map(pd.Timestamp.toordinal).to_numpy()
    chunks = ['<!doctype html><html><head><meta charset="utf-8"><title>EIA result review</title><style>body{font:16px Arial;color:#1f2937;margin:28px auto;max-width:1200px}svg{max-width:900px;width:100%}table{border-collapse:collapse;font-size:12px}th,td{padding:6px;border:1px solid #ddd}.scroll{overflow:auto}pre{white-space:pre-wrap}</style></head><body><h1>EIA saved-result review</h1><p>No recalibration. Red markers indicate optimizer nonconvergence. Connected lines join observations.</p>']
    chunks += ['<h2>Annual diagnostics</h2>',table(by_year.reset_index())]
    for param in PARAMS+["RMSE_USD"]:
        color = '#2563eb'
        chunks += [f'<h2>{param}</h2>',chart([(param,ordinal,results[param].to_numpy(),color)],
                    'Date',param,marks=list(zip(ordinal[~results.converged],results.loc[~results.converged,param])))]
    chunks += ['<h2>Monthly other-bound percentage</h2>',
               chart([('Other bounds (%)',monthly.index.to_timestamp().map(pd.Timestamp.toordinal).to_numpy(),monthly.other_bound_pct.to_numpy(),'#d97706')],'Date','% of observed dates')]
    chunks += ['<h2>All 18 nonconverged dates</h2><div class="scroll">',table(failures[columns]),'</div>']
    selected = pd.concat([worst.head(5).Date,failures.head(3).Date]).drop_duplicates()
    for day in selected:
        group = prices[prices.Date==day].sort_values('Maturity')
        x = (group.ExpiryDate-group.Date).dt.days.to_numpy()
        chunks += [f'<h2>Price comparison: {day.date()}</h2>',
                   chart([('Observed',x,group.Price.to_numpy(),'#d97706'),
                          ('Saved fitted',x,group.FittedPrice.to_numpy(),'#2563eb')],
                         'Days to last trade','USD/barrel'),
                   table(group[["Contract","ExpiryDate","Maturity","Price","FittedPrice","Error_USD"]])]
    chunks += ['<h2>Review notes</h2><pre>',html.escape('\n'.join(report)),'</pre></body></html>']
    (OUT / "result_review.html").write_text(''.join(chunks),encoding='utf-8')
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in hashes.items())
    assert len(pd.read_csv(OUT / "nonconverged_dates.csv"))==18
    print('\n'.join(report),flush=True)


if __name__ == "__main__":
    main()
