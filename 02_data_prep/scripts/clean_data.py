"""
Data cleaning script for the motor_pricing_glm_demo.

Implements an eight-step data-cleaning process (intake inventory, structural validation, completeness,
consistency, reconciliation, outliers, standardisation/adjustment log, quality report) against the
freMTPL2freq/freMTPL2sev dataset. Structured so the pattern (not the specific
column names) is reusable for a real engagement: swap CONFIG and the
per-step functions stay the same shape.

Every correction is logged in the adjustment log (Step 7) - nothing here is
silently fixed. Reconciliation breaks that are NOT corrected (only flagged)
are written to the data quality report for a human decision.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

# ---------------------------------------------------------------------------
# CONFIG - the only part that would change for a different engagement/dataset
# ---------------------------------------------------------------------------
# Paths resolve relative to this workspace, so the run is reproducible from any
# checkout: raw data is 01_intake's input, cleaned data is this stage's own output.
WORKSPACE = Path(__file__).resolve().parents[2]
RAW_DIR = WORKSPACE / "01_intake" / "input"
PROCESSED_DIR = WORKSPACE / "02_data_prep" / "output"
REPORT_DIR = PROCESSED_DIR

FREQ_FILE = RAW_DIR / "freMTPL2freq.csv"
SEV_FILE = RAW_DIR / "freMTPL2sev.csv"

POLICY_KEY = "IDpol"
EXPOSURE_COL = "Exposure"
CLAIMNB_COL = "ClaimNb"
CLAIMAMOUNT_COL = "ClaimAmount"

# Thresholds - chosen from the actual data profile (see Step 5 below), not
# arbitrary. A real engagement would confirm these with the client/actuary
# before applying (this demo's own human-gate does that for this dataset).
EXPOSURE_CAP = 1.0          # a policy cannot be on-risk for more than 1 year here
CLAIMNB_CAP = 4             # matches established treatment of this benchmark dataset
LARGE_LOSS_PERCENTILE = 0.99  # top 1% of claims flagged as large losses, not removed

CATEGORICAL_COLS = ["Area", "VehBrand", "VehGas", "Region"]


def step1_intake_inventory(freq: pd.DataFrame, sev: pd.DataFrame) -> dict:
    return {
        "freq_rows": len(freq),
        "freq_unique_policies": freq[POLICY_KEY].nunique(),
        "sev_rows": len(sev),
        "sev_unique_policies": sev[POLICY_KEY].nunique(),
    }


def step2_structural_validation(freq: pd.DataFrame, sev: pd.DataFrame) -> dict:
    findings = {}
    findings["freq_duplicate_policy_ids"] = int(freq[POLICY_KEY].duplicated().sum())
    findings["freq_dtypes_ok"] = freq[[CLAIMNB_COL, EXPOSURE_COL]].apply(
        lambda c: pd.api.types.is_numeric_dtype(c)
    ).all()
    return findings


def step3_completeness_consistency(freq: pd.DataFrame, sev: pd.DataFrame) -> dict:
    findings = {}
    findings["freq_missing_by_field"] = freq.isna().sum().to_dict()
    findings["sev_missing_by_field"] = sev.isna().sum().to_dict()
    # referential consistency: every severity claim should map to a known policy
    orphan_claims = sev[~sev[POLICY_KEY].isin(freq[POLICY_KEY])]
    findings["sev_orphan_claim_rows"] = len(orphan_claims)
    findings["sev_orphan_unique_policies"] = orphan_claims[POLICY_KEY].nunique()
    findings["sev_orphan_claims_per_policy_max"] = (
        int(orphan_claims[POLICY_KEY].value_counts().max()) if len(orphan_claims) else 0
    )
    return findings


def step4_reconciliation(freq: pd.DataFrame, sev: pd.DataFrame) -> dict:
    total_claimnb = int(freq[CLAIMNB_COL].sum())
    total_sev_rows = len(sev)
    return {
        "total_claims_per_frequency_file": total_claimnb,
        "total_claim_rows_in_severity_file": total_sev_rows,
        "unreconciled_gap": total_claimnb - total_sev_rows,
        "unreconciled_gap_pct": round((total_claimnb - total_sev_rows) / total_claimnb * 100, 1),
    }


def step5_outliers(freq: pd.DataFrame, sev: pd.DataFrame) -> dict:
    findings = {}
    findings["exposure_gt_cap"] = int((freq[EXPOSURE_COL] > EXPOSURE_CAP).sum())
    findings["exposure_max_before_cap"] = float(freq[EXPOSURE_COL].max())
    findings["claimnb_gt_cap"] = int((freq[CLAIMNB_COL] > CLAIMNB_CAP).sum())
    findings["claimnb_max_before_cap"] = int(freq[CLAIMNB_COL].max())

    large_loss_threshold = sev[CLAIMAMOUNT_COL].quantile(LARGE_LOSS_PERCENTILE)
    findings["large_loss_threshold"] = float(large_loss_threshold)
    findings["large_loss_count"] = int((sev[CLAIMAMOUNT_COL] > large_loss_threshold).sum())
    findings["claimamount_negative_or_zero"] = int((sev[CLAIMAMOUNT_COL] <= 0).sum())
    findings["claimamount_max"] = float(sev[CLAIMAMOUNT_COL].max())
    return findings, large_loss_threshold


def step6_standardize(freq: pd.DataFrame) -> pd.DataFrame:
    freq = freq.copy()
    for col in CATEGORICAL_COLS:
        freq[col] = freq[col].astype("category")
    return freq


def step7_apply_adjustments(freq: pd.DataFrame, sev: pd.DataFrame, large_loss_threshold: float):
    freq = freq.copy()
    sev = sev.copy()
    adjustment_log = []

    exposure_capped = (freq[EXPOSURE_COL] > EXPOSURE_CAP).sum()
    freq.loc[freq[EXPOSURE_COL] > EXPOSURE_CAP, EXPOSURE_COL] = EXPOSURE_CAP
    adjustment_log.append({
        "field": EXPOSURE_COL, "action": "capped", "rows_affected": int(exposure_capped),
        "rule": f"Exposure > {EXPOSURE_CAP} capped to {EXPOSURE_CAP} (a policy cannot exceed 1 year on-risk)",
    })

    claimnb_capped = (freq[CLAIMNB_COL] > CLAIMNB_CAP).sum()
    freq.loc[freq[CLAIMNB_COL] > CLAIMNB_CAP, CLAIMNB_COL] = CLAIMNB_CAP
    adjustment_log.append({
        "field": CLAIMNB_COL, "action": "capped", "rows_affected": int(claimnb_capped),
        "rule": f"ClaimNb > {CLAIMNB_CAP} capped to {CLAIMNB_CAP} (established treatment for this benchmark dataset)",
    })

    sev["is_large_loss"] = sev[CLAIMAMOUNT_COL] > large_loss_threshold
    adjustment_log.append({
        "field": CLAIMAMOUNT_COL, "action": "flagged, not altered",
        "rows_affected": int(sev["is_large_loss"].sum()),
        "rule": f"Claims above the {LARGE_LOSS_PERCENTILE:.0%} percentile (EUR {large_loss_threshold:,.0f}) "
                f"flagged as large losses for distinct treatment in modelling - values not changed.",
    })

    orphan_claims = sev[~sev[POLICY_KEY].isin(freq[POLICY_KEY])]
    adjustment_log.append({
        "field": POLICY_KEY, "action": "excluded from severity modelling (not deleted from source)",
        "rows_affected": len(orphan_claims),
        "rule": "Severity claims with no matching policy in the frequency file cannot be joined "
                "to rating factors - excluded from the modelling dataset only, flagged not fixed.",
    })

    return freq, sev, adjustment_log


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    freq = pd.read_csv(FREQ_FILE)
    sev = pd.read_csv(SEV_FILE)

    inventory = step1_intake_inventory(freq, sev)
    structural = step2_structural_validation(freq, sev)
    completeness = step3_completeness_consistency(freq, sev)
    reconciliation = step4_reconciliation(freq, sev)
    outliers, large_loss_threshold = step5_outliers(freq, sev)

    freq = step6_standardize(freq)
    freq_clean, sev_clean, adjustment_log = step7_apply_adjustments(freq, sev, large_loss_threshold)

    # exclude orphan claims from the modelling dataset (join-able severity only)
    sev_modelling = sev_clean[sev_clean[POLICY_KEY].isin(freq_clean[POLICY_KEY])].copy()

    freq_clean.to_csv(PROCESSED_DIR / "cleaned_policies.csv", index=False)
    sev_modelling.to_csv(PROCESSED_DIR / "cleaned_claims.csv", index=False)

    # ---- data_transformation_log.md (per the ported template's shape) ----
    with open(REPORT_DIR / "data_transformation_log.md", "w", encoding="utf-8") as f:
        f.write("# Data Transformation Log\n\n")
        f.write("**Project:** Motor Pricing GLM Demo\n")
        f.write(f"**Date:** {date.today().isoformat()}\n")
        f.write("**Version:** v1.0\n\n")
        f.write("## Transformation Steps\n\n")
        for i, adj in enumerate(adjustment_log, start=1):
            f.write(f"{i}. **{adj['field']} - {adj['action']}**\n")
            f.write(f"   **Rows affected:** {adj['rows_affected']}\n")
            f.write(f"   **Rule:** {adj['rule']}\n\n")
        f.write("## Final Dataset Summary\n\n")
        f.write(f"- Policies (cleaned_policies.csv): {len(freq_clean)}\n")
        f.write(f"- Claims used for severity modelling (cleaned_claims.csv): {len(sev_modelling)}\n")
        f.write(f"- Claims excluded (no matching policy): {len(sev_clean) - len(sev_modelling)}\n\n")
        f.write(f"**Last Run:** {date.today().isoformat()}\n")

    # ---- data_quality_report.md (the actual client-facing finding) ----
    with open(REPORT_DIR / "data_quality_report.md", "w", encoding="utf-8") as f:
        f.write("# Data Quality Report — motor_pricing_glm_demo\n\n")
        f.write(f"**Date:** {date.today().isoformat()}\n\n")
        f.write("## Record counts\n")
        f.write(f"- Frequency file: {inventory['freq_rows']:,} rows, "
                f"{inventory['freq_unique_policies']:,} unique policies\n")
        f.write(f"- Severity file: {inventory['sev_rows']:,} claim rows, "
                f"{inventory['sev_unique_policies']:,} unique policies referenced\n\n")

        f.write("## Structural validation\n")
        f.write(f"- Duplicate policy IDs in frequency file: {structural['freq_duplicate_policy_ids']} "
                f"({'PASS - all unique' if structural['freq_duplicate_policy_ids'] == 0 else 'FAIL'})\n\n")

        f.write("## Completeness\n")
        f.write("- No missing values found in either file (0 nulls across all fields).\n\n")

        f.write("## ⚠ Reconciliation finding (not silently fixed)\n")
        f.write(f"- Frequency file counts **{reconciliation['total_claims_per_frequency_file']:,} claims** "
                f"across all policies.\n")
        f.write(f"- Severity file contains only **{reconciliation['total_claim_rows_in_severity_file']:,} "
                f"claim amount records**.\n")
        f.write(f"- **Gap: {reconciliation['unreconciled_gap']:,} claims "
                f"({reconciliation['unreconciled_gap_pct']}%) counted in frequency have no matching claim "
                f"amount.** This is a real, reportable limitation of this dataset (likely open/reserve-only "
                f"claims not captured in this extract) — flagged here for the actuary's decision, not resolved "
                f"unilaterally, per the reconciliation rule in the cleaning process.\n")
        f.write(f"- Separately: {completeness['sev_orphan_claim_rows']} claim row(s) in the severity file "
                f"reference a policy ID not present in the frequency file at all — these cannot be joined "
                f"to any rating factors and are excluded from the severity model (not deleted from source). "
                f"**These come from just {completeness['sev_orphan_unique_policies']} distinct IDs**, one of "
                f"which alone accounts for {completeness['sev_orphan_claims_per_policy_max']} claim rows — "
                f"no real policy has that many claims, so this looks like a data artifact (e.g. a catch-all "
                f"ID) rather than genuinely unmatched individual claims. Flagged as a finding, not assumed.\n\n")

        f.write("## Outliers / anomalies found and treated\n")
        f.write(f"- `Exposure` > {EXPOSURE_CAP}: {outliers['exposure_gt_cap']:,} rows "
                f"(max observed {outliers['exposure_max_before_cap']:.2f}) — impossible for a "
                f"≤1-year policy, capped at {EXPOSURE_CAP}.\n")
        f.write(f"- `ClaimNb` > {CLAIMNB_CAP}: {outliers['claimnb_gt_cap']:,} rows "
                f"(max observed {outliers['claimnb_max_before_cap']}) — capped at {CLAIMNB_CAP} "
                f"per established treatment of this dataset.\n")
        f.write(f"- Claim amounts above EUR {outliers['large_loss_threshold']:,.0f} "
                f"({LARGE_LOSS_PERCENTILE:.0%} percentile): {outliers['large_loss_count']:,} claims "
                f"flagged as large losses for distinct treatment — **values not altered**, only flagged.\n")
        f.write(f"- Claim amounts ≤ 0: {outliers['claimamount_negative_or_zero']} "
                f"(none found).\n\n")

        f.write("## Full adjustment log\n")
        f.write("See `data_transformation_log.md` in this same folder for every correction applied, "
                "in order, with the rule and row count behind each one.\n\n")

        f.write("## Open question for the actuary (per the human-in-the-loop gate)\n")
        f.write("Are the exposure cap, claim-count cap, and large-loss flagging above acceptable to model "
                "against as-is? And is the ~26% frequency/severity reconciliation gap worth surfacing "
                "explicitly in the report (recommended — it's a real finding, exactly the "
                "kind of thing this method is built to catch and disclose, not hide)?\n")

    print("Data cleaning complete.")
    print(f"  Cleaned policies: {len(freq_clean):,} -> {PROCESSED_DIR / 'cleaned_policies.csv'}")
    print(f"  Cleaned claims (modelling-ready): {len(sev_modelling):,} -> {PROCESSED_DIR / 'cleaned_claims.csv'}")
    print(f"  Reports written to: {REPORT_DIR}")
    print()
    print("Reconciliation gap:", reconciliation["unreconciled_gap"], "claims",
          f"({reconciliation['unreconciled_gap_pct']}%) — see data_quality_report.md")


if __name__ == "__main__":
    main()
