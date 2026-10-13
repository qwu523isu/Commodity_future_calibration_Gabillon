"""Prepare EIA nearby CL prices with rule-derived expiries; fit 3 case dates.

Run directly. Raw inputs and previous calibration outputs are preserved.
Calendar scope is deliberately restricted to 2020-2024. Expiries are inferred,
with historical exchange-notice spot checks, not a downloaded security master.
"""
from pathlib import Path
from datetime import date, timedelta
import hashlib
import html
import numpy as np
import pandas as pd
from Step1_Calibrate_Gabillon_model_parameters import (
    PARAM_NAMES, bound_flags, calibrate_for_date, gabillon_forward,
)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Input_data" / "EIA_WTI_futures_2020-01-02_to_2024-04-05.csv"
OUT = ROOT / "Output" / "Step1a_EIA_Preparation"
MONTH_CODES = "FGHJKMNQUVXZ"
CASE_DATES = ["2020-03-16", "2022-03-08", "2024-04-05"]
EIA_URL = "https://www.eia.gov/dnav/pet/TblDefs/pet_pri_fut_tbldef2.asp"
RULE_URL = "https://www.cmegroup.com/rulebook/NYMEX/2/200.pdf"
# Only holiday dates that can affect this rule's 18th-25th counting window
# are needed. These are not a generic exchange trading-session calendar.
HOLIDAYS = {date.fromisoformat(s) for s in [
    "2020-01-20", "2020-05-25", "2020-11-26", "2020-12-25",
    "2021-01-18", "2021-11-25", "2021-12-24",
    "2022-01-17", "2022-06-20", "2022-11-24", "2022-12-26",
    "2023-01-16", "2023-06-19", "2023-11-23", "2023-12-25",
    "2024-01-15", "2024-06-19", "2024-11-28", "2024-12-25",
]}
# Delivery month -> independently published CL last trade and source.
CHECKS = {
    "2020-04": ("2020-03-20", "https://www.cmegroup.com/notices/clearing/2020/03/Chadv20-098.pdf"),
    "2020-05": ("2020-04-21", "https://www.cmegroup.com/notices/clearing/2020/04/Chadv20-161Revised.pdf"),
    "2020-06": ("2020-05-19", "https://www.cmegroup.com/notices/clearing/2020/05/Chadv20-199.pdf"),
    "2021-01": ("2020-12-21", "https://www.cmegroup.com/notices/clearing/2020/12/Chadv20-489.pdf"),
    "2021-04": ("2021-03-22", "https://www.cmegroup.com/notices/clearing/2021/03/Chadv21-092.pdf"),
    "2021-12": ("2021-11-19", "https://www.cmegroup.com/notices/clearing/2021/11/Chadv21-426.pdf"),
    "2022-04": ("2022-03-22", "https://www.cmegroup.com/notices/clearing/2022/03/Chadv22-108.pdf"),
    "2022-12": ("2022-11-21", "https://www.cmegroup.com/notices/clearing/2022/11/Chadv22-448.pdf"),
    "2023-04": ("2023-03-21", "https://www.cmegroup.com/notices/clearing/2023/03/Chadv23-097.pdf"),
    "2023-12": ("2023-11-20", "https://www.cmegroup.com/notices/clearing/2023/11/Chadv23-348.pdf"),
    "2024-04": ("2024-03-20", "https://www.cmegroup.com/notices/clearing/2024/03/Chadv24-083.pdf"),
    "2024-05": ("2024-04-22", "https://www.cmegroup.com/content/dam/cmegroup/notices/clearing/2024/04/chadv24-113.pdf"),
    "2024-07": ("2024-06-20", "https://www.cmegroup.com/notices/clearing/2024/06/Chadv24-177.pdf"),
}


def expiry(delivery_month):
    prior = delivery_month - 1
    d = date(prior.year, prior.month, 25)
    if not 2020 <= d.year <= 2024:
        raise ValueError("Expiry calendar supports 2020-2024 only")
    def business_day(day):
        return day.weekday() < 5 and day not in HOLIDAYS
    while not business_day(d):
        d -= timedelta(days=1)
    for _ in range(3):
        d -= timedelta(days=1)
        while not business_day(d):
            d -= timedelta(days=1)
    return d


def front_month(day):
    month = pd.Period(day, freq="M") + 1
    if day > expiry(month):  # keep expiring contract on its last trading day
        month += 1
    return month


def prepare():
    raw = pd.read_csv(SOURCE, parse_dates=["Date"])
    assert len(raw) == 1070 and raw.Date.is_unique and raw.Date.is_monotonic_increasing
    assert raw.Date.min() == pd.Timestamp("2020-01-02")
    assert raw.Date.max() == pd.Timestamp("2024-04-05")
    values = raw[[f"Contract{i}" for i in range(1, 5)]].to_numpy(float)
    assert np.isfinite(values).all()
    for month, (expected, _) in CHECKS.items():
        assert expiry(pd.Period(month)) == date.fromisoformat(expected), month
    assert str(front_month(date(2020, 4, 21))) == "2020-05"
    assert str(front_month(date(2020, 4, 22))) == "2020-06"
    assert str(front_month(date(2020, 5, 19))) == "2020-06"
    assert str(front_month(date(2020, 5, 20))) == "2020-07"
    assert str(front_month(date(2020, 12, 22))) == "2021-02"
    records = []
    for _, row in raw.iterrows():
        day = row.Date.date()
        front = front_month(day)
        eligible = all(row[f"Contract{i}"] > 0 for i in range(1, 5))
        for rank in range(1, 5):
            month = front + rank - 1
            last_trade = expiry(month)
            days = (last_trade - day).days
            assert days >= 0
            check = CHECKS.get(str(month))
            records.append({
                "Date": day, "NearbyRank": rank,
                "Contract": f"CL{MONTH_CODES[month.month-1]}{month.year}",
                "DeliveryMonth": str(month), "ExpiryDate": last_trade,
                "DaysToExpiry": days, "Maturity": days / 365.0,
                "Price": row[f"Contract{rank}"], "CalibrationEligible": eligible,
                "ExclusionReason": "" if eligible else "nonpositive_price_on_date",
                "ExpiryMethod": "CME_rule_200102F_inferred",
                "ExpiryNoticeChecked": check is not None,
                "ExpiryNoticeSource": check[1] if check else "",
            })
    long = pd.DataFrame(records)
    assert len(long) == 4280
    assert long.groupby("Date").size().eq(4).all()
    assert not long.duplicated(["Date", "Contract"]).any()
    assert long.groupby("Date").Maturity.apply(lambda x: x.is_monotonic_increasing).all()
    assert long.loc[~long.CalibrationEligible, "Date"].nunique() == 1
    assert str(long.loc[~long.CalibrationEligible, "Date"].iloc[0]) == "2020-04-20"
    return raw, long


def plot_svg(group, params):
    t = group.Maturity.to_numpy()
    market = group.Price.to_numpy()
    grid = np.linspace(t.min(), t.max(), 150)
    fitted = np.exp(gabillon_forward(grid, *params))
    lo, hi = min(market.min(), fitted.min()), max(market.max(), fitted.max())
    pad = max((hi-lo)*.15, .1)
    lo, hi = lo-pad, hi+pad
    x = lambda v: 65 + (v-t.min())/(t.max()-t.min())*490
    y = lambda v: 270 - (v-lo)/(hi-lo)*230
    points = " ".join(f"{x(a):.2f},{y(b):.2f}" for a,b in zip(grid,fitted))
    parts = ['<svg viewBox="0 0 600 330" role="img" aria-label="Observed and fitted futures curve">',
             '<rect width="600" height="330" fill="white"/>',
             '<path d="M65 40 V270 H555" fill="none" stroke="#555"/>',
             f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/>']
    for a,b in zip(t,market):
        parts.append(f'<circle cx="{x(a):.2f}" cy="{y(b):.2f}" r="5" fill="#d97706"/>')
        parts.append(f'<text x="{x(a):.2f}" y="292" text-anchor="middle" font-size="12">{a*365:.0f}</text>')
    for v in np.linspace(lo,hi,5):
        parts.append(f'<text x="58" y="{y(v)+4:.2f}" text-anchor="end" font-size="12">{v:.2f}</text>')
    parts += ['<text x="300" y="320" text-anchor="middle" font-size="14">Days to last trade</text>',
              '<text x="65" y="25" font-size="14">USD/barrel; orange: observed, blue: fitted</text>', '</svg>']
    return "".join(parts)


def main():
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    raw, long = prepare()
    OUT.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT / "EIA_WTI_contract_details.csv", index=False)
    usable = long[long.CalibrationEligible]
    usable.to_csv(OUT / "EIA_WTI_calibration_input.csv", index=False)
    long[~long.CalibrationEligible].to_csv(OUT / "excluded_date_details.csv", index=False)
    long[["Contract", "DeliveryMonth", "ExpiryDate", "ExpiryMethod", "ExpiryNoticeChecked",
          "ExpiryNoticeSource"]].drop_duplicates().sort_values("DeliveryMonth").to_csv(
              OUT / "CL_expiry_calendar.csv", index=False)
    results, comparisons, charts = [], [], []
    for day in CASE_DATES:
        group = usable[usable.Date == date.fromisoformat(day)].sort_values("Maturity")
        assert len(group) == 4
        res, bounds = calibrate_for_date(group)
        fitted = np.exp(gabillon_forward(group.Maturity.to_numpy(), *res.x))
        flags = bound_flags(res.x, bounds)
        results.append({"Date": day, **dict(zip(PARAM_NAMES, res.x)),
                        "converged": bool(res.success), "optimizer_message": str(res.message),
                        "RMSE_USD": np.sqrt(np.mean((fitted-group.Price.to_numpy())**2)),
                        **{f"on_bound_{p}": flag for p,flag in zip(PARAM_NAMES,flags)}})
        comparison = group.copy()
        comparison["FittedPrice"] = fitted
        comparison["Error_USD"] = fitted-group.Price.to_numpy()
        comparisons.append(comparison)
        charts.append(f"<h2>{day}</h2>"+plot_svg(group,res.x)+
                      comparison[["Contract","ExpiryDate","DaysToExpiry","Price","FittedPrice",
                                  "Error_USD"]].to_html(index=False,float_format=lambda v:f"{v:.4f}"))
    result = pd.DataFrame(results)
    result.to_csv(OUT / "sample_calibration_results.csv", index=False)
    pd.concat(comparisons).to_csv(OUT / "sample_price_comparison.csv", index=False)
    notes = [
        "EIA WTI preparation and 3-date curve-fit check",
        f"Source: {SOURCE.name}; SHA256: {before}",
        "Original data preserved. No interpolated prices and no synthetic long tenors.",
        f"Raw dates: {len(raw)}; all-contract rows: {len(long)}.",
        f"Calibration dates: {usable.Date.nunique()}; rows: {len(usable)}.",
        "2020-04-20 excluded as a whole date because Contract1 is -37.63.",
        "Contract1 maps to earliest unexpired delivery month; roll occurs after last trade.",
        "Maturity = actual calendar days to last trading date / 365 (ACT/365F).",
        "Intraday timing ignored; on-expiry observations retained with maturity=0.",
        "Expiry dates reconstructed from CME rule 200102F and holidays in its counting window.",
        f"Independent exchange-notice checks: {len(CHECKS)} delivery months passed.",
        "Other delivery months are rule-derived, not independently checked against a security master.",
        "EIA contract identities are inferred from its published nearby convention.",
        "Physical delivery spans a month; last trade is the maturity proxy for this curve-fit study.",
        "Four short-dated points do not validate long-term parameters or out-of-sample accuracy.",
        "This is a sample-date experiment, not a full-history calibration or simulation fix.",
        f"EIA definitions: {EIA_URL}", f"CME expiry rule: {RULE_URL}", "",
        result.to_string(index=False),
    ]
    (OUT / "preparation_report.txt").write_text("\n".join(notes),encoding="utf-8")
    page = '<!doctype html><html><head><meta charset="utf-8"><title>EIA WTI sample fits</title><style>body{font:16px Arial;max-width:1000px;margin:32px auto;color:#1f2937}svg{max-width:650px}table{border-collapse:collapse;margin:16px 0 36px}th,td{padding:8px;border:1px solid #ddd}pre{white-space:pre-wrap}</style></head><body><h1>EIA WTI sample curve fits</h1><p>Expiries inferred from CME rules; ACT/365F. Four nearby prices per date. In-sample fits only.</p>'
    page += result.to_html(index=False,float_format=lambda v:f"{v:.6f}")+"".join(charts)
    page += "<h2>Preparation notes</h2><pre>"+html.escape("\n".join(notes))+"</pre></body></html>"
    (OUT / "sample_curve_fits.html").write_text(page,encoding="utf-8")
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == before
    assert len(pd.read_csv(OUT / "EIA_WTI_calibration_input.csv")) == 4276
    print("\n".join(notes))


if __name__ == "__main__":
    main()
