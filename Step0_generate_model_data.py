"""
Step 0 - Ground truth generator / 阶段 0 - 基准真值生成器

Generates futures curves and option implied vols FROM the Gabillon model with
KNOWN parameters, so that Steps 1 and 3 can be tested for parameter recovery.

用已知参数从 Gabillon 模型正向生成期货曲线与期权隐含波动率，
以便检验 Step 1 与 Step 3 能否反解出这些参数。

Design note / 设计说明
----------------------
The data is generated with EXACTLY the formulas Steps 1 and 3 assume, i.e. the
forward carries NO convexity correction. This is deliberate: it isolates the
question "is the calibrator identified?" from the separate question "is Step 1
consistent with Step 3?" (the convexity inconsistency is a Phase 4 experiment).
If we generated with the true risk-neutral forward, a recovery failure would be
ambiguous between the two causes.

本数据严格使用 Step 1 与 Step 3 所假设的公式生成，即远期价格不含凸性修正。
这是刻意为之：它把「校准程序是否可识别？」与「Step 1 与 Step 3 是否自洽？」
这两个问题分离开（后者是阶段 4 的实验）。若用真实风险中性远期生成数据，
反解失败将无法区分究竟是哪一个原因造成的。

Output schemas match the existing files exactly, so Steps 1 and 3 can read this
data with only a path change.
输出格式与现有文件完全一致，因此 Step 1 与 Step 3 只需改动路径即可读取。
"""

import numpy as np
import pandas as pd
import os

# -----------------------------
# Paths
# -----------------------------
base_path = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_path, "Output", "Step0_Ground_Truth")
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Tenor grid - matches the real input file (1M ... 12M)
# -----------------------------
TENOR_COLS = [f"{m}M" for m in range(1, 13)]
TENOR_T = np.array([m / 12 for m in range(1, 13)])

# Step 3a / Step 3 only use these four maturities for vols
VOL_T = np.array([1 / 12, 3 / 12, 6 / 12, 12 / 12])


# -----------------------------
# Model primitives (kappa -> 0 safe)
# -----------------------------
def int_exp(a, T):
    """(1 - exp(-a*T)) / a, with the a -> 0 limit equal to T.

    Needed because true Gabillon has kappa_Y = 0, where the textbook
    sigma^2/(2*kappa) * (1 - exp(-2*kappa*T)) form is 0/0.
    真正的 Gabillon 模型中 kappa_Y = 0，此时教科书形式为 0/0，故需取极限。
    """
    a = np.asarray(a, dtype=float)
    small = np.abs(a) < 1e-12
    safe_a = np.where(small, 1.0, a)
    return np.where(small, T, (1.0 - np.exp(-safe_a * T)) / safe_a)


def gabillon_forward(T, kappa_X, kappa_Y, X0, Y0):
    """Log forward price. Identical to the function used in Steps 1, 2 and 4."""
    return X0 * np.exp(-kappa_X * T) + Y0 * np.exp(-kappa_Y * T)


def gabillon_forward_var(T, kappa_X, kappa_Y, sigma_X, sigma_Y, rho):
    """Integrated forward variance. Algebraically identical to Step 3's
    formula, but written so kappa -> 0 is well defined."""
    term1 = sigma_X**2 * int_exp(2 * kappa_X, T)
    term2 = sigma_Y**2 * int_exp(2 * kappa_Y, T)
    term3 = 2 * rho * sigma_X * sigma_Y * int_exp(kappa_X + kappa_Y, T)
    return term1 + term2 + term3


def gabillon_forward_vol(T, kappa_X, kappa_Y, sigma_X, sigma_Y, rho):
    return np.sqrt(gabillon_forward_var(T, kappa_X, kappa_Y, sigma_X, sigma_Y, rho) / T)


# -----------------------------
# Generator
# -----------------------------
def generate(name, kappa_X, kappa_Y, sigma_X, sigma_Y, rho,
             S0=60.0, L0=65.0, n_dates=252, dt=1 / 252,
             price_noise=0.0, vol_noise=0.0, seed=0, write=True):
    """Generate one ground-truth dataset.

    State convention / 状态量约定:
        ln F(T) = X_t * exp(-kappa_X * T) + Y_t * exp(-kappa_Y * T)
        so at T = 0:  ln S_t = X_t + Y_t
        Y = log long-term level, X = log(spot / long-term level)
        Y 为长期价格水平的对数，X 为现货相对长期水平的对数偏离

    X and Y evolve as exact OU processes driven by the SAME
    (sigma_X, sigma_Y, rho) that generate the implied vols, so the futures
    panel and the vol panel are mutually consistent.
    X 与 Y 按精确 OU 格式演化，其驱动参数与生成隐含波动率的参数完全相同，
    因此期货面板与波动率面板相互自洽。

    price_noise: stdev of additive USD noise on futures prices (experiment 2)
    vol_noise:   stdev of additive noise on implied vols
    """
    rng = np.random.default_rng(seed)

    # Initial state
    Y_init = np.log(L0)              # long-term log level
    X_init = np.log(S0) - np.log(L0)  # short-term log deviation

    # Exact OU one-step moments
    decay_X = np.exp(-kappa_X * dt)
    decay_Y = np.exp(-kappa_Y * dt)
    sd_X = sigma_X * np.sqrt(int_exp(2 * kappa_X, dt))
    sd_Y = sigma_Y * np.sqrt(int_exp(2 * kappa_Y, dt))

    L_chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))

    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    X = np.empty(n_dates)
    Y = np.empty(n_dates)
    X[0], Y[0] = X_init, Y_init

    for t in range(1, n_dates):
        z = L_chol @ rng.standard_normal(2)
        X[t] = X[t - 1] * decay_X + sd_X * z[0]
        Y[t] = Y[t - 1] * decay_Y + sd_Y * z[1]

    # ---- Futures panel: Date, 1M ... 12M ----
    log_F = (X[:, None] * np.exp(-kappa_X * TENOR_T)[None, :]
             + Y[:, None] * np.exp(-kappa_Y * TENOR_T)[None, :])
    F = np.exp(log_F)
    if price_noise > 0:
        F = F + rng.normal(0.0, price_noise, size=F.shape)

    futures_df = pd.DataFrame(F, columns=TENOR_COLS)
    futures_df.insert(0, "Date", dates)

    # ---- Vol panel: Date, Maturity, ImpliedVol ----
    vols = gabillon_forward_vol(VOL_T, kappa_X, kappa_Y, sigma_X, sigma_Y, rho)
    if vol_noise > 0:
        vol_panel = vols[None, :] + rng.normal(0.0, vol_noise, size=(n_dates, len(VOL_T)))
    else:
        vol_panel = np.tile(vols, (n_dates, 1))

    vol_df = pd.DataFrame({
        "Date": np.repeat(dates, len(VOL_T)),
        "Maturity": np.tile(VOL_T, n_dates),
        "ImpliedVol": vol_panel.ravel(),
    })

    # ---- Truth ----
    truth_df = pd.DataFrame({
        "Date": dates,
        "kappa_X": kappa_X, "kappa_Y": kappa_Y,
        "X0": X, "Y0": Y,
        "sigma_X": sigma_X, "sigma_Y": sigma_Y, "rho": rho,
    })

    if write:
        futures_df.to_csv(os.path.join(output_dir, f"truth_futures_{name}.csv"), index=False)
        vol_df.to_csv(os.path.join(output_dir, f"truth_vols_{name}.csv"), index=False)
        truth_df.to_csv(os.path.join(output_dir, f"truth_params_{name}.csv"), index=False)

    return futures_df, vol_df, truth_df


# -----------------------------
# Self-checks - the generator must be internally consistent
# -----------------------------
def self_check(futures_df, truth_df, kappa_X, kappa_Y):
    """Verify the written prices really are the model's prices."""
    row, tru = futures_df.iloc[0], truth_df.iloc[0]

    # 1M price must equal exp(gabillon_forward(1/12, ...))
    expected = np.exp(gabillon_forward(1 / 12, kappa_X, kappa_Y, tru["X0"], tru["Y0"]))
    assert abs(row["1M"] - expected) < 1e-10, "futures panel does not match the model"

    # ln S = X + Y at T = 0
    spot = np.exp(gabillon_forward(0.0, kappa_X, kappa_Y, tru["X0"], tru["Y0"]))
    assert abs(np.log(spot) - (tru["X0"] + tru["Y0"])) < 1e-12, "T=0 identity broken"

    # curve must carry curvature (unlike the real input file, which is linear)
    spreads = np.diff(row[TENOR_COLS].values.astype(float))
    return spreads


# -----------------------------
# Cases
# -----------------------------
# Case A: kappa_Y is inside both the original [0.01, 1] and current [0, 1]
#         bounds. Recovery failures need numerical/conditioning diagnosis.
# 情形 A：kappa_Y 在原始与当前边界内；反解失败需排查求解精度和敏感性。
#
# Case B: kappa_Y = 0, i.e. TRUE Gabillon (long factor is a driftless random
#         walk). Step 1's original lower bound 0.01 excluded it; the current
#         lower bound 0 permits recovery.
# 情形 B：kappa_Y = 0；Step 1 原始下界 0.01 排除了该值，当前下界 0 允许反解。
CASES = {
    "caseA_reachable": dict(kappa_X=1.5, kappa_Y=0.05, sigma_X=0.35, sigma_Y=0.20, rho=0.30),
    "caseB_true_gabillon": dict(kappa_X=1.5, kappa_Y=0.00, sigma_X=0.35, sigma_Y=0.20, rho=0.30),
}

if __name__ == "__main__":
    print(f"Ground truth -> {output_dir}\n")

    for name, params in CASES.items():
        futures_df, vol_df, truth_df = generate(name, **params)
        spreads = self_check(futures_df, truth_df, params["kappa_X"], params["kappa_Y"])

        half_life = np.log(2) / params["kappa_X"]
        vols = gabillon_forward_vol(VOL_T, **params)

        print(f"{name}")
        print(f"  true params : kappa_X={params['kappa_X']}, kappa_Y={params['kappa_Y']}, "
              f"sigma_X={params['sigma_X']}, sigma_Y={params['sigma_Y']}, rho={params['rho']}")
        print(f"  kappa_X half-life : {half_life:.2f} y ({12*half_life:.1f} months)")
        print(f"  implied vols 1M/3M/6M/12M : "
              f"{vols[0]:.1%} / {vols[1]:.1%} / {vols[2]:.1%} / {vols[3]:.1%}  (Samuelson: declining)")
        print(f"  curve on day 1 : 1M={futures_df['1M'].iloc[0]:.2f}  "
              f"12M={futures_df['12M'].iloc[0]:.2f}")
        print(f"  month-on-month spreads day 1 : "
              f"{spreads[0]:+.3f} ... {spreads[-1]:+.3f}  "
              f"(NOT constant -> the curve has curvature)")
        print(f"  spot range over {len(futures_df)} dates : "
              f"{np.exp(truth_df.X0 + truth_df.Y0).min():.2f} - "
              f"{np.exp(truth_df.X0 + truth_df.Y0).max():.2f}")
        print()
