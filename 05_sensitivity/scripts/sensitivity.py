"""
Sensitivity analysis script for the motor_pricing_glm_demo.

Implements the sensitivity step of a standard pricing process: vary the assumptions
flagged as most judgment-heavy, show the range of outcomes, not just the
central estimate. Scores from the coefficients in 04_model/output/glm_spec.json -
does not refit anything.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = Path(__file__).resolve().parents[2]
MODEL_DIR = WORKSPACE / "04_model" / "output"
OUTPUT_DIR = WORKSPACE / "05_sensitivity" / "output"

BASE_PROFILE = {"VehPower": 6, "VehAge": 6, "DrivAge": 44, "BonusMalus": 50,
                "VehGas": "Regular", "Area": "C"}

BONUS_MALUS_RANGE = [50, 70, 90, 110, 130, 150, 200]
DRIV_AGE_RANGE = [18, 20, 23, 25, 28, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]


def apply_bands(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    df = df.copy()
    for factor, b in spec["bands"].items():
        df[f"{factor}_band"] = pd.cut(df[factor], bins=b["edges"], labels=b["labels"]).astype(str)
    return df


def predict_log(params: dict, df: pd.DataFrame, spec: dict) -> np.ndarray:
    """Same scorer as 04_model/scripts/pricing_glm.py:predict_log."""
    eta = np.full(len(df), params["Intercept"])
    for f in spec["rating_factors"]:
        base = spec["base_levels"][f]
        eta = eta + df[f].map(lambda lvl: params.get(f"C({f}, Treatment(reference='{base}'))[T.{lvl}]", 0.0)).to_numpy()
    return eta


def technical_premium(spec: dict, df: pd.DataFrame) -> np.ndarray:
    df = apply_bands(df, spec)
    # BonusMalus enters as the statutory BM/100 (an offset in the frequency GLM).
    freq = np.exp(predict_log(spec["freq_params"], df, spec) + np.log(df["BonusMalus"] / 100.0))
    sev = np.exp(predict_log(spec["sev_params"], df, spec)) * spec["large_loss_load"]
    return freq * sev / (1 - spec["expense_loading"] - spec["profit_margin"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = json.loads((MODEL_DIR / "glm_spec.json").read_text(encoding="utf-8"))

    bm_df = pd.DataFrame([{**BASE_PROFILE, "BonusMalus": bm} for bm in BONUS_MALUS_RANGE])
    bm_df["technical_premium"] = technical_premium(spec, bm_df)
    # What the data would say if the slope were estimated freely (04_model challenger GLM).
    beta = spec["bonus_malus_challenger"]["freq_slope"]
    ref_at_100 = technical_premium(spec, pd.DataFrame([{**BASE_PROFILE, "BonusMalus": 100}]))[0]
    bm_df["challenger_free_slope_premium"] = ref_at_100 * (bm_df["BonusMalus"].clip(upper=150) / 100.0) ** beta
    bm_df.to_csv(OUTPUT_DIR / "sensitivity_bonus_malus.csv", index=False)

    age_df = pd.DataFrame([{**BASE_PROFILE, "DrivAge": age} for age in DRIV_AGE_RANGE])
    age_df["technical_premium"] = technical_premium(spec, age_df)
    age_df.to_csv(OUTPUT_DIR / "sensitivity_driver_age.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(bm_df["BonusMalus"], bm_df["technical_premium"], marker="o", color="#003366",
                 label="Productie: wettelijke BM/100")
    axes[0].plot(bm_df["BonusMalus"], bm_df["challenger_free_slope_premium"], marker="s", ls="--",
                 color="#FF6600", label=f"Challenger: vrije helling ({beta:.2f}), cap 150")
    axes[0].set_xlabel("Bonus-malus")
    axes[0].set_ylabel("Technische premie (illustratief, EUR)")
    axes[0].set_title("Gevoeligheid: bonus-malus")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Plotted per year of age so the class boundaries (right-inclusive) sit where they really are.
    dense = pd.DataFrame([{**BASE_PROFILE, "DrivAge": a} for a in range(18, 91)])
    axes[1].step(dense["DrivAge"], technical_premium(spec, dense), where="mid", color="#008080")
    axes[1].plot(age_df["DrivAge"], age_df["technical_premium"], "o", color="#008080")
    axes[1].set_xlabel("Leeftijd bestuurder")
    axes[1].set_title("Gevoeligheid: leeftijd bestuurder (klassen)")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Technische premie — basisprofiel: gemiddelde polis (illustratieve opslagen)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sensitivity_chart.png", dpi=150)

    print("Sensitivity analysis complete.")
    print(bm_df[["BonusMalus", "technical_premium", "challenger_free_slope_premium"]].round(1).to_string(index=False))
    print(f"  Bonus-malus range: EUR {bm_df.technical_premium.min():.0f} - {bm_df.technical_premium.max():.0f} "
          f"({bm_df.technical_premium.max() / bm_df.technical_premium.min():.1f}x)")
    print(age_df[["DrivAge", "technical_premium"]].round(1).to_string(index=False))
    print(f"  Driver-age range: EUR {age_df.technical_premium.min():.0f} - {age_df.technical_premium.max():.0f} "
          f"({age_df.technical_premium.max() / age_df.technical_premium.min():.1f}x)")
    print(f"  Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
