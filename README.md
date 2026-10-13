# Commodity futures calibration and risk engine

A Python research project for WTI futures curve calibration and a minimal batch portfolio risk engine. The current risk engine is an **architecture demonstration using synthetic volatility**, not a validated market risk forecasting system.

## Run the risk demo

Use Python 3.12 from the repository root:

```sh
python -m pip install -r requirements-risk.txt
python run_risk_engine.py
python -m unittest discover -s tests -v
```

Edit `Input_data/risk_engine_config.json` to change net positions, scenario count, confidence, seed and synthetic volatility. Positive quantities are long; negative quantities are short. Supply one net position per contract.

The demo supports the reviewed valuation/horizon pair **2024-04-05 to 2024-04-08** only. It reads committed market and calibration snapshots; it does not rerun calibration. Outputs go to `Output/Step4b_Portfolio_Risk_Engine` by default. Use `--config` and `--output` to select other files/directories. Output directories are overwritten when reused; select a new directory to retain a previous run.

## Implemented architecture

```text
EIA data -> cleaning / contract expiry mapping -> curve calibration snapshots
                                                        |
position and scenario configuration --------------------+
                                                        v
shared two-factor scenarios -> futures revaluation -> portfolio P&L -> VaR / ES
```

`Step1a_prepare_EIA_data.py` prepares the EIA sample. `Step1c_calibrate_EIA_full_sample.py` calibrates the daily four-parameter curves. `Step4a_minimal_risk_engine.py` demonstrates a single position; `Step4b_portfolio_risk_engine.py` adds shared-factor portfolio scenarios. `run_risk_engine.py` is the batch entry point and returns a nonzero exit code on failure.

Risk assumptions: ACT/365F calendar time, zero expected futures price return, illustrative factor volatilities and correlation, and no fees, liquidity or delivery modelling. Calibration convergence and bound flags are retained. Curve parameters can still hit their bounds. These limitations are not resolved by passing software tests.

## CI

`.github/workflows/risk-engine-ci.yml` runs on pushes, pull requests and manual dispatch. It installs the pinned risk dependencies on Ubuntu / Python 3.12 and runs four tests, including a 1,000-scenario CLI integration run. Tests cover valuation direction and scaling, known tail risk, configuration rejection, portfolio aggregation, reproducibility and missing quotes. No credentials or live market feed are needed.

The workflow needs these committed input snapshots:

- `Output/Step1a_EIA_Preparation/EIA_WTI_calibration_input.csv`
- `Output/Step1c_EIA_Full_Sample/EIA_calibration_results.csv`

Other generated outputs are ignored. Requiring the `Python 3.12 risk tests` check before merging is a separate GitHub branch protection/ruleset setting.

## Data and research scripts

The EIA sample covers 2020-01-02 to 2024-04-05. The negative front-month observation on 2020-04-20 is preserved in the extracted raw sample; that date is excluded from logarithmic calibration. Calendar mapping uses a last-trading-day proxy and a limited, explicitly coded holiday set, not a general exchange calendar.

The original root-level historical CSV is retained. A copy under `Input_data` supports the relative paths in the legacy scripts. Research diagnostics are included but have dependencies beyond `requirements-risk.txt` (for example SciPy for optimization, and Matplotlib for legacy plots). The original spot simulation is not the portfolio risk entry point. Prior GitHub scripts remain in Git history; local manual backup folders are not committed.

## Planned, not deployed

An API, asynchronous worker/queue, persistent task records, server deployment and CD remain future work. Current execution is local batch processing; GitHub Actions provides CI once this branch is pushed and Actions is enabled.
