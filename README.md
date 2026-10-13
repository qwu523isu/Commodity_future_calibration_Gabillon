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

`.github/workflows/risk-engine-ci.yml` runs on pushes, pull requests and manual dispatch. It installs the pinned risk dependencies on Ubuntu / Python 3.12 and runs six tests, including CLI integration and real local HTTP requests. Tests cover valuation direction and scaling, known tail risk, configuration rejection, portfolio aggregation, reproducibility, missing quotes, API submission and result retrieval. No credentials or live market feed are needed.

The workflow needs these committed input snapshots:

- `Output/Step1a_EIA_Preparation/EIA_WTI_calibration_input.csv`
- `Output/Step1c_EIA_Full_Sample/EIA_calibration_results.csv`

Other generated outputs are ignored. Requiring the `Python 3.12 risk tests` check before merging is a separate GitHub branch protection/ruleset setting.

## Data and research scripts

The EIA sample covers 2020-01-02 to 2024-04-05. The negative front-month observation on 2020-04-20 is preserved in the extracted raw sample; that date is excluded from logarithmic calibration. Calendar mapping uses a last-trading-day proxy and a limited, explicitly coded holiday set, not a general exchange calendar.

The original root-level historical CSV is retained. A copy under `Input_data` supports the relative paths in the legacy scripts. Research diagnostics are included but have dependencies beyond `requirements-risk.txt` (for example SciPy for optimization, and Matplotlib for legacy plots). The original spot simulation is not the portfolio risk entry point. Prior GitHub scripts remain in Git history; local manual backup folders are not committed.

## Local API prototype

Start from the repository root:

```sh
python risk_api.py
```

The server defaults to `127.0.0.1:8000`. It uses the Python standard library and adds no dependencies. Routes:

| Method | Route | Behaviour |
| --- | --- | --- |
| GET | `/health` | Health and architecture-demo label |
| POST | `/runs` | Accept the configuration JSON, calculate synchronously, return HTTP 201 with run ID and summary |
| GET | `/runs/{run_id}` | Read a previously completed result; unknown or failed runs return 404 |

Example from another terminal (PowerShell users should use `curl.exe`):

```sh
curl http://127.0.0.1:8000/health
curl -H "Content-Type: application/json" --data-binary @Input_data/risk_engine_config.json http://127.0.0.1:8000/runs
```

Each request saves its configuration and outputs in a separate `Output/API_Runs/{run_id}` directory. Invalid inputs return 400; internal calculation failures return 500. The API limits scenario counts to one million. The current single-process server handles one request at a time: POST waits for calculation and other requests wait too. Failed requests can leave a request configuration for debugging, but no completed result is published. There is no background queue, task-status lifecycle or authentication. This local server is not suitable for public deployment.

## Docker

From the repository root, on a machine with Docker:

```sh
docker build -t risk-api:local .
docker run -d --name risk-api-local -p 127.0.0.1:8000:8000 -v risk-api-results:/data risk-api:local
curl http://127.0.0.1:8000/health
python scripts/smoke_risk_api.py --state-file smoke-result.json
docker logs risk-api-local
docker stop risk-api-local
```

The image includes only the risk API, two engine modules, configuration and required data snapshots. Python 3.12 and pinned NumPy/Pandas dependencies run as user 10001. Inside the container the API explicitly binds to `0.0.0.0`; the example publishes the port only on host loopback. A named volume at `/data` retains run results after container replacement. Do not mount over `/app/Output`, which contains the required input snapshots. Local bind mounts require appropriate write permissions for user 10001.

CI first runs the six Python tests, then builds the Docker image and checks the actual container over HTTP. It submits a 1,000-scenario risk run, retrieves the result, replaces the container while retaining its named volume, and retrieves the same result again. Images are built for verification, not published to a registry or deployed to a server. The Python base image tag is maintained upstream; it is not pinned to an immutable digest.

Docker port publishing reference: [Docker documentation](https://docs.docker.com/engine/network/port-publishing/).

## Planned, not deployed

An asynchronous worker/queue, persistent task-status records, production server deployment and CD remain future work. Current execution supports local batch processing and a synchronous API prototype packaged in Docker; GitHub Actions verifies tests and container behaviour. This remains a single-request-at-a-time demonstration server with synthetic volatility, not a production API.
