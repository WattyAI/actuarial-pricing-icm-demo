"""
ML challenger for the motor_pricing_glm_demo (frequency only).

The GLM in 04_model stays the production model. This stage runs gradient
boosting next to it as a diagnostic - the "GLM + ML, not GLM or ML" loop:

  1. Refit the production GLM on a training split (same spec as 04_model).
  2. Fit two boosted models (scikit-learn HistGradientBoosting, Poisson loss,
     exposure handled as rate target + exposure weight - equivalent to an
     exposure offset):
       a. a standalone GBM challenger, with a monotone-increasing constraint
          on BonusMalus (a worse bonus-malus step never lowers the frequency);
       b. a correction model boosted on top of the GLM prediction: it can only
          learn what the GLM misses.
  3. Compare out-of-sample Poisson deviance on the holdout split.
  4. Explain both boosted models with exact interventional Shapley values
     (log scale, so SHAP values read as log-relativities) and pairwise Shapley
     interaction indices. Computed directly - no shap/lightgbm dependency.
  5. Write candidate signals for the actuary. Nothing is promoted into the GLM
     by this script: that is the human gate in CONTEXT.md.

Severity is out of scope here: it explains <1% of deviance in 04_model, so a
challenger has little to find and the demo stays focused on frequency.
"""
import json
from itertools import combinations
from math import factorial
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = Path(__file__).resolve().parents[2]
DATA_DIR = WORKSPACE / "02_data_prep" / "output"
MODEL_DIR = WORKSPACE / "04_model" / "output"
OUTPUT_DIR = WORKSPACE / "04_model_challenger" / "output"

SEED = 2026
TEST_SHARE = 0.2
FEATURES = ["DrivAge", "VehAge", "VehPower", "BonusMalus", "Area", "VehGas"]
CATEGORICAL = ["Area", "VehGas"]
N_EXPLAIN, N_BACKGROUND = 1500, 150
# The run-2 GLM structure (linear continuous terms), kept as a baseline so the
# gain from banding is visible next to the gain from boosting.
LINEAR_FORMULA = "ClaimNb ~ VehPower + VehAge + DrivAge + C(VehGas) + C(Area)"


def apply_bands(df, spec):
    df = df.copy()
    for factor, b in spec["bands"].items():
        df[f"{factor}_band"] = pd.cut(df[factor], bins=b["edges"], labels=b["labels"]).astype(str)
    return df


def poisson_deviance(y, mu):
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(2 * np.sum(term - (y - mu)))


def encode(df):
    X = df[FEATURES].copy()
    X["Area"] = X["Area"].map({a: i for i, a in enumerate("ABCDEF")})
    X["VehGas"] = (X["VehGas"] == "Regular").astype(int)
    return X.to_numpy(dtype=float)


def gbm(monotone=None, leaf=31, min_leaf=1000):
    return HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=600, max_leaf_nodes=leaf,
        min_samples_leaf=min_leaf, l2_regularization=1.0,
        categorical_features=[FEATURES.index(c) for c in CATEGORICAL],
        monotonic_cst=monotone, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=SEED)


def shapley(predict_log, X_explain, X_background):
    """Exact interventional Shapley values and pairwise interaction indices.

    v(S)(x) = mean_b predict_log(x_S, b_notS). With p=6 features that is 64
    coalitions - small enough to enumerate exactly rather than approximate."""
    n, p = X_explain.shape
    m = len(X_background)
    v = np.empty((1 << p, n))
    bg = np.tile(X_background, (n, 1))
    xs = np.repeat(X_explain, m, axis=0)
    for mask in range(1 << p):
        Z = bg.copy()
        cols = [j for j in range(p) if mask >> j & 1]
        Z[:, cols] = xs[:, cols]
        v[mask] = predict_log(Z).reshape(n, m).mean(axis=1)

    phi = np.zeros((n, p))
    for i in range(p):
        for mask in range(1 << p):
            if mask >> i & 1:
                continue
            s = bin(mask).count("1")
            w = factorial(s) * factorial(p - s - 1) / factorial(p)
            phi[:, i] += w * (v[mask | 1 << i] - v[mask])

    inter = {}
    for i, j in combinations(range(p), 2):
        acc = np.zeros(n)
        for mask in range(1 << p):
            if mask >> i & 1 or mask >> j & 1:
                continue
            s = bin(mask).count("1")
            w = factorial(s) * factorial(p - s - 2) / factorial(p - 1)
            acc += w * (v[mask | 1 << i | 1 << j] - v[mask | 1 << i] - v[mask | 1 << j] + v[mask])
        inter[(FEATURES[i], FEATURES[j])] = acc
    efficiency_gap = float(np.max(np.abs(phi.sum(axis=1) - (v[-1] - v[0]))))
    return phi, inter, efficiency_gap


def md(df, index=False):
    """Markdown table without the optional tabulate dependency."""
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    body = ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def ae_table(df, by, mu_col):
    g = df.groupby(by, observed=True).agg(exposure=("Exposure", "sum"), actual=("ClaimNb", "sum"),
                                          expected=(mu_col, "sum"))
    g["A_over_E"] = g["actual"] / g["expected"]
    return g.round(3)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = json.loads((MODEL_DIR / "glm_spec.json").read_text(encoding="utf-8"))
    df = pd.read_csv(DATA_DIR / "cleaned_policies.csv")
    df = apply_bands(df[df["Exposure"] > 0], spec).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    test_mask = rng.random(len(df)) < TEST_SHARE
    train, test = df[~test_mask].copy(), df[test_mask].copy()
    bm_off = lambda d: np.log(d["BonusMalus"] / 100.0)
    exp_off = lambda d: np.log(d["Exposure"])

    # ---- 1. GLMs on the training split ----
    print("Refitting GLMs on the training split...")
    glm_prod = smf.glm(spec["freq_formula"], data=train, family=sm.families.Poisson(),
                       offset=exp_off(train) + bm_off(train)).fit()
    glm_lin = smf.glm(LINEAR_FORMULA, data=train, family=sm.families.Poisson(),
                      offset=exp_off(train) + bm_off(train)).fit()
    for d in (train, test):
        d["mu_glm"] = glm_prod.predict(d, offset=exp_off(d) + bm_off(d))
        d["mu_glm_linear"] = glm_lin.predict(d, offset=exp_off(d) + bm_off(d))

    # ---- 2. boosted models ----
    Xtr, Xte = encode(train), encode(test)
    mono = [0] * len(FEATURES)
    mono[FEATURES.index("BonusMalus")] = 1
    print("Fitting standalone GBM challenger...")
    g_std = gbm(monotone=mono).fit(Xtr, train["ClaimNb"] / train["Exposure"], sample_weight=train["Exposure"])
    print("Fitting GBM correction on top of the GLM...")
    w_tr = train["mu_glm"].to_numpy()
    g_cor = gbm(leaf=15, min_leaf=2000).fit(Xtr, train["ClaimNb"] / w_tr, sample_weight=w_tr)
    test["mu_gbm"] = g_std.predict(Xte) * test["Exposure"]
    test["mu_glm_x_correction"] = test["mu_glm"] * g_cor.predict(Xte)

    # ---- 3. out-of-sample comparison ----
    y = test["ClaimNb"]
    null_mu = test["Exposure"] * train["ClaimNb"].sum() / train["Exposure"].sum()
    d_null = poisson_deviance(y, null_mu)
    rows = []
    for name, col, desc in [
        ("GLM, linear terms (run-2 structure)", "mu_glm_linear", "5 factors, continuous age/power terms, BM offset"),
        ("GLM, banded (production)", "mu_glm", "5 banded factors, BM offset"),
        ("GBM challenger (standalone)", "mu_gbm", f"{g_std.n_iter_} trees, BM monotone increasing"),
        ("GLM x GBM correction", "mu_glm_x_correction", f"production GLM + {g_cor.n_iter_}-tree correction"),
    ]:
        d = poisson_deviance(y, test[col])
        rows.append({"model": name, "description": desc, "holdout_poisson_deviance": round(d, 1),
                     "deviance_explained_vs_null": round(1 - d / d_null, 5)})
    comp = pd.DataFrame(rows)
    d_prod = comp.loc[comp.model == "GLM, banded (production)", "holdout_poisson_deviance"].iloc[0]
    comp["deviance_reduction_vs_production"] = ((d_prod - comp["holdout_poisson_deviance"]) / d_prod).round(5)
    comp.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    # ---- 4. Shapley values ----
    print("Computing exact Shapley values (64 coalitions per model)...")
    ex = Xte[rng.choice(len(Xte), N_EXPLAIN, replace=False)]
    bgd = Xtr[rng.choice(len(Xtr), N_BACKGROUND, replace=False)]
    phi_std, inter_std, gap_std = shapley(lambda Z: np.log(g_std.predict(Z)), ex, bgd)
    phi_cor, inter_cor, gap_cor = shapley(lambda Z: np.log(g_cor.predict(Z)), ex, bgd)
    imp = pd.DataFrame({"feature": FEATURES,
                        "gbm_mean_abs_shap": np.abs(phi_std).mean(axis=0).round(4),
                        "correction_mean_abs_shap": np.abs(phi_cor).mean(axis=0).round(4)})
    imp = imp.sort_values("correction_mean_abs_shap", ascending=False)
    imp.to_csv(OUTPUT_DIR / "shap_importance.csv", index=False)
    inter = pd.DataFrame([{"pair": f"{a} x {b}", "gbm_mean_abs_interaction": round(float(np.abs(inter_std[(a, b)]).mean()), 4),
                           "correction_mean_abs_interaction": round(float(np.abs(inter_cor[(a, b)]).mean()), 4)}
                          for a, b in inter_std]).sort_values("correction_mean_abs_interaction", ascending=False)
    inter.to_csv(OUTPUT_DIR / "shap_interactions.csv", index=False)

    # ---- dependence plots: driver age and bonus-malus ----
    rel = pd.read_csv(MODEL_DIR / "relativities.csv")
    age_rel = rel[rel.factor == "DrivAge"].set_index("level")["freq_relativity"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ia, ib = FEATURES.index("DrivAge"), FEATURES.index("BonusMalus")
    axes[0].scatter(ex[:, ia], phi_std[:, ia], s=6, alpha=0.35, color="#008080", label="GBM: SHAP-waarde (log)")
    labels = spec["bands"]["DrivAge"]["labels"]
    edges = spec["bands"]["DrivAge"]["edges"]
    # Centre the GLM log-relativities on the explained sample, like SHAP values are.
    lr = np.log(age_rel[labels].to_numpy())
    band_idx = np.searchsorted(edges, ex[:, ia], side="left") - 1
    lr = lr - lr[band_idx].mean()
    for (lo, hi), val in zip(zip(edges[:-1], edges[1:]), lr):
        axes[0].hlines(val, max(lo, 18), min(hi, 90), color="#003366", lw=2.5)
    axes[0].plot([], [], color="#003366", lw=2.5, label="GLM: klasse-relativiteit (log, gecentreerd)")
    axes[0].set_xlabel("Leeftijd bestuurder")
    axes[0].set_ylabel("Bijdrage aan log-frequentie")
    axes[0].set_title("Leeftijd: GBM ontdekt de vorm, GLM volgt met klassen")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].scatter(ex[:, ib], phi_cor[:, ib], s=6, alpha=0.35, color="#FF6600")
    axes[1].axhline(0, color="grey", lw=1)
    axes[1].set_xlabel("Bonus-malus")
    axes[1].set_ylabel("Correctie op het GLM (log)")
    axes[1].set_title("Wat het GLM mist: correctie-SHAP per bonus-malus")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "shap_dependence.png", dpi=150)

    # ---- 5. candidate signals (evidence only - the actuary decides) ----
    top_main = imp.head(2)["feature"].tolist()
    top_pair = inter.iloc[0]["pair"].split(" x ")
    full = pd.concat([train, test])
    full["bm_band"] = pd.cut(full["BonusMalus"], [0, 50, 60, 80, 100, 125, 999],
                             labels=["50", "51-60", "61-80", "81-100", "101-125", "126+"])
    full["bm_100_plus"] = np.where(full["BonusMalus"] >= 100, "BM>=100", "BM<100")
    groupers = {"BonusMalus": "bm_band", "DrivAge": "DrivAge_band", "VehAge": "VehAge_band",
                "VehPower": "VehPower_band", "Area": "Area", "VehGas": "VehGas"}
    ae_main = {f: ae_table(full, groupers[f], "mu_glm") for f in top_main}
    pair_groupers = [groupers[f] if f != "BonusMalus" else "bm_100_plus" for f in top_pair]
    ae_pair = ae_table(full, pair_groupers, "mu_glm")

    summary = {"seed": SEED, "test_share": TEST_SHARE, "n_train": int((~test_mask).sum()),
               "n_test": int(test_mask.sum()), "gbm_trees": int(g_std.n_iter_),
               "correction_trees": int(g_cor.n_iter_), "shap_explained_points": N_EXPLAIN,
               "shap_background_points": N_BACKGROUND,
               "shap_efficiency_gap_max": {"gbm": gap_std, "correction": gap_cor},
               "top_correction_features": top_main, "top_correction_interaction": top_pair}
    (OUTPUT_DIR / "challenger_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with open(OUTPUT_DIR / "candidate_signals.md", "w", encoding="utf-8") as fh:
        fh.write("# Candidate signals — ML challenger (generated, not a decision)\n\n")
        fh.write(f"Generated by `04_model_challenger/scripts/gbm_challenger.py`, seed {SEED}, "
                 f"{int(test_mask.sum()):,}-policy holdout. Every figure below is evidence for the "
                 "actuary's gate in `../CONTEXT.md`; nothing here is promoted into the GLM by the script.\n\n")
        fh.write("## Out-of-sample comparison (frequency, Poisson deviance)\n\n")
        fh.write(md(comp) + "\n\n")
        fh.write("## What the correction model learns on top of the GLM (mean |SHAP|, log scale)\n\n")
        fh.write(md(imp) + "\n\n")
        fh.write("## Strongest pairwise interactions (Shapley interaction index)\n\n")
        fh.write(md(inter.head(5)) + "\n\n")
        for f, t in ae_main.items():
            fh.write(f"## Production GLM actual/expected by {f} (full data)\n\n{md(t, index=True)}\n\n")
        fh.write(f"## Production GLM actual/expected by {top_pair[0]} x {top_pair[1]} (full data)\n\n"
                 f"{md(ae_pair, index=True)}\n\n")
    print(comp.to_string(index=False))
    print(imp.to_string(index=False))
    print(inter.head(6).to_string(index=False))
    for f, t in ae_main.items():
        print(f"\nA/E by {f}\n{t}")
    print(f"\nA/E by {top_pair}\n{ae_pair}")
    print(f"\nShapley efficiency gap (should be ~0): {gap_std:.2e}, {gap_cor:.2e}")


if __name__ == "__main__":
    main()
