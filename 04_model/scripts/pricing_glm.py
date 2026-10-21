"""
Pricing model script for the actuarial pricing ICM workflow.

With `--config`, fits a configurable frequency GLM and, optionally, a severity
GLM to prepared data using explicitly mapped columns. The no-argument
invocation remains the French Motor TPL reference implementation for
reproducibility; its fixed variables and assumptions are not generic defaults.

Generic configuration example: `04_model/model_config.example.json`.
Read `04_model/CONTEXT.md` and obtain the data-preparation sign-off before
fitting either mode.
"""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


def _read_generic_config(config_path: Path) -> dict:
    """Load a generic-run JSON config and resolve file paths from workspace root."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The model configuration must be a JSON object.")
    required = {"policy_data", "policy_columns", "features"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(sorted(missing))}")
    if not isinstance(config["policy_data"], str) or not config["policy_data"]:
        raise ValueError("policy_data must be a non-empty path string.")
    if not isinstance(config["policy_columns"], dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value
        for key, value in config["policy_columns"].items()
    ):
        raise ValueError("policy_columns must map standard fields to source column names.")

    for key in ("policy_data", "claims_data", "output_dir"):
        value = config.get(key)
        if value:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a path string.")
            path = Path(value)
            config[key] = path if path.is_absolute() else WORKSPACE / path
    config["config_path"] = config_path
    return config


def _formula_term(column: str, categorical: bool) -> str:
    """Quote arbitrary source column names safely in a Patsy formula."""
    quoted = f'Q({json.dumps(column)})'
    return f"C({quoted})" if categorical else quoted


def _formula_terms(features: dict) -> list[str]:
    if not isinstance(features, dict):
        raise ValueError("features must be an object with categorical/numeric lists.")
    categorical = features.get("categorical", [])
    numeric = features.get("numeric", [])
    if (not isinstance(categorical, list) or not isinstance(numeric, list)
            or not all(isinstance(value, str) and value for value in categorical + numeric)):
        raise ValueError("features.categorical and features.numeric must be lists.")
    if len(set(categorical + numeric)) != len(categorical) + len(numeric):
        raise ValueError("A feature cannot be listed more than once.")
    return ([_formula_term(column, True) for column in categorical]
            + [_formula_term(column, False) for column in numeric])


def _validate_numeric(series: pd.Series, name: str, *, positive: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError(f"Mapped field '{name}' must contain finite numeric values.")
    if positive and (values <= 0).any():
        raise ValueError(f"Mapped field '{name}' must be strictly positive.")
    return values


def _fit_frequency(data: pd.DataFrame, count_column: str, exposure_column: str,
                   terms: list[str], family_name: str, alpha: float | None = None):
    response = _formula_term(count_column, False)
    formula = response + (" ~ " + " + ".join(terms) if terms else " ~ 1")
    if not isinstance(family_name, str):
        raise ValueError("frequency.family must be a string.")
    family_name = family_name.lower()
    if family_name == "poisson":
        family = sm.families.Poisson()
    elif family_name == "negative_binomial":
        if alpha is None or not np.isfinite(float(alpha)) or float(alpha) <= 0:
            raise ValueError(
                "frequency.alpha must be a finite positive value for "
                "negative_binomial."
            )
        family = sm.families.NegativeBinomial(alpha=float(alpha))
    else:
        raise ValueError("frequency.family must be 'poisson' or 'negative_binomial'.")
    model = smf.glm(
        formula, data=data, family=family,
        offset=np.log(data[exposure_column]),
    ).fit()
    return formula, model


def _fit_optional_severity(config: dict, policies: pd.DataFrame,
                           features: list[str], feature_terms: list[str]):
    claims_path = config.get("claims_data")
    if not claims_path:
        return None

    policy_columns = config["policy_columns"]
    claim_columns = config.get("claim_columns", {})
    if not isinstance(claim_columns, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value
        for key, value in claim_columns.items()
    ):
        raise ValueError("claim_columns must map standard fields to source column names.")
    required = {"policy_id", "period", "amount"}
    missing = required - claim_columns.keys()
    if missing:
        raise ValueError(f"claim_columns missing: {', '.join(sorted(missing))}")
    if not policy_columns.get("policy_id") or not policy_columns.get("period"):
        raise ValueError(
            "Claims require policy_columns.policy_id and policy_columns.period "
            "to align claims to policy-period records."
        )

    claims = pd.read_csv(claims_path)
    policy_id = policy_columns["policy_id"]
    policy_period = policy_columns["period"]
    claim_id = claim_columns["policy_id"]
    claim_period = claim_columns["period"]
    amount_column = claim_columns["amount"]
    for frame, columns in ((policies, [policy_id, policy_period] + features),
                           (claims, [claim_id, claim_period, amount_column])):
        absent = [column for column in columns if column not in frame.columns]
        if absent:
            raise ValueError(f"Mapped columns not found: {', '.join(absent)}")

    policies = policies.copy()
    claims = claims.copy()
    policies["__period_key"] = policies[policy_period].astype("string").str.strip()
    claims["__period_key"] = claims[claim_period].astype("string").str.strip()
    policies["__policy_key"] = policies[policy_id].astype("string").str.strip()
    claims["__policy_key"] = claims[claim_id].astype("string").str.strip()
    if (policies["__period_key"].isna().any()
            or claims["__period_key"].isna().any()
            or policies["__period_key"].eq("").any()
            or claims["__period_key"].eq("").any()
            or policies["__policy_key"].isna().any()
            or claims["__policy_key"].isna().any()
            or policies["__policy_key"].eq("").any()
            or claims["__policy_key"].eq("").any()):
        raise ValueError("Policy identifiers and period mappings cannot be missing.")

    policy_keys = ["__policy_key", "__period_key"]
    if policies.duplicated(policy_keys).any():
        raise ValueError(
            "Policy data must be unique by policy ID and period before joining claims."
        )
    policy_periods = set(policies["__period_key"].unique())
    claim_periods = set(claims["__period_key"].unique())
    common_periods = sorted(policy_periods & claim_periods)
    if not common_periods:
        raise ValueError("Policy and claims data have no common periods.")
    excluded_policy_periods = sorted(policy_periods - set(common_periods))
    excluded_claim_periods = sorted(claim_periods - set(common_periods))
    policy_rows_before_alignment = len(policies)
    claim_rows_before_alignment = len(claims)
    policies = policies[policies["__period_key"].isin(common_periods)].copy()
    claims = claims[claims["__period_key"].isin(common_periods)].copy()

    policy_features = policies[policy_keys + features]
    joined = claims.merge(
        policy_features,
        left_on=["__policy_key", "__period_key"],
        right_on=policy_keys,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched_claims = int((joined["_merge"] != "both").sum())
    joined = joined[joined["_merge"] == "both"].drop(
        columns=["_merge", "__period_key", "__policy_key"]
    )
    if joined.empty:
        raise ValueError(
            "No claim records match policy IDs and periods in the common periods."
        )
    amount_name = "__severity_amount"
    joined[amount_name] = _validate_numeric(joined[amount_column], amount_column)
    if (joined[amount_name] <= 0).any():
        raise ValueError(
            "Gamma severity modelling requires positive claim amounts; "
            "resolve non-positive values through approved preparation."
        )

    severity_config = config.get("severity", {})
    if not isinstance(severity_config, dict):
        raise ValueError("severity must be an object containing model settings.")
    cap = severity_config.get("cap")
    if cap is not None:
        cap = float(cap)
        if not np.isfinite(cap) or cap <= 0:
            raise ValueError("severity.cap must be a finite positive number or null.")
        joined[amount_name] = joined[amount_name].clip(upper=cap)
    family_name = severity_config.get("family", "gamma")
    if not isinstance(family_name, str):
        raise ValueError("severity.family must be a string.")
    if not isinstance(family_name, str):
        raise ValueError("frequency.family must be a string.")
    family_name = family_name.lower()
    if family_name != "gamma":
        raise ValueError("severity.family currently supports 'gamma' only.")
    response = _formula_term(amount_name, False)
    formula = response + (" ~ " + " + ".join(feature_terms) if feature_terms else " ~ 1")
    model = smf.glm(
        formula, data=joined,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit()
    return {
        "model": model,
        "formula": formula,
        "policies": policies,
        "claims": joined,
        "amount_column": amount_column,
        "amount_name": amount_name,
        "cap": cap,
        "common_periods": common_periods,
        "excluded_policy_periods": excluded_policy_periods,
        "excluded_claim_periods": excluded_claim_periods,
        "excluded_policy_rows": policy_rows_before_alignment - len(policies),
        "excluded_claim_rows": claim_rows_before_alignment - int(
            claims["__period_key"].isin(common_periods).sum()
        ),
        "unmatched_claims": unmatched_claims,
        "policy_id": policy_id,
        "claim_id": claim_id,
        "claim_period": claim_period,
    }


def main_generic(config_path: Path) -> None:
    """Fit mapped frequency data and optional period-aligned severity data."""
    config = _read_generic_config(config_path)
    policy_columns = config["policy_columns"]
    if not isinstance(policy_columns, dict):
        raise ValueError("policy_columns must be an object of standard-to-source mappings.")
    required = {"period", "exposure", "claim_count"}
    missing = required - policy_columns.keys()
    if missing:
        raise ValueError(f"policy_columns missing: {', '.join(sorted(missing))}")

    policies = pd.read_csv(config["policy_data"])
    if not isinstance(config["features"], dict):
        raise ValueError("features must be an object with categorical/numeric lists.")
    terms = _formula_terms(config["features"])
    exposure_column = policy_columns["exposure"]
    count_column = policy_columns["claim_count"]
    period_column = policy_columns["period"]
    features = config["features"].get("categorical", []) + config["features"].get("numeric", [])
    forbidden_features = {
        period_column, exposure_column, count_column, policy_columns.get("policy_id")
    }
    invalid_features = sorted((set(features) & forbidden_features) - {None})
    if invalid_features:
        raise ValueError(
            "Identifiers, period, exposure, and claim count cannot also be "
            f"predictors: {', '.join(invalid_features)}."
        )
    mapped = [period_column, exposure_column, count_column] + features
    if policy_columns.get("policy_id"):
        mapped.append(policy_columns["policy_id"])
    absent = [column for column in mapped if column not in policies.columns]
    if absent:
        raise ValueError(f"Mapped columns not found in policy data: {', '.join(absent)}")
    policies = policies[mapped].copy()
    policies[exposure_column] = _validate_numeric(
        policies[exposure_column], exposure_column, positive=True
    )
    policies[count_column] = _validate_numeric(policies[count_column], count_column)
    if (policies[count_column] < 0).any() or not np.equal(
        policies[count_column], np.floor(policies[count_column])
    ).all():
        raise ValueError(f"Mapped claim count '{count_column}' must be non-negative integers.")
    if policies[mapped].isna().any().any():
        missing_fields = policies[mapped].columns[policies[mapped].isna().any()].tolist()
        raise ValueError(
            "Missing values in mapped fields require an approved preparation "
            f"decision: {', '.join(missing_fields)}."
        )
    for column in config["features"].get("numeric", []):
        policies[column] = _validate_numeric(policies[column], column)
    if policies.empty:
        raise ValueError("No policy records remain after validating mapped features.")

    severity = _fit_optional_severity(config, policies, features, terms)
    if severity:
        policies = severity["policies"]
    frequency_config = config.get("frequency", {})
    if not isinstance(frequency_config, dict):
        raise ValueError("frequency must be an object containing model settings.")
    freq_formula, freq_model = _fit_frequency(
        policies, count_column, exposure_column, terms,
        frequency_config.get("family", "poisson"),
        frequency_config.get("alpha"),
    )

    output_dir = config.get("output_dir", WORKSPACE / "04_model" / "output" / "generic")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generic_frequency_summary.txt").write_text(
        f"Formula: {freq_formula}\n"
        f"Family: {frequency_config.get('family', 'poisson')}\n"
        f"Records: {len(policies):,}\n"
        f"Periods: {', '.join(sorted(policies[period_column].astype(str).unique()))}\n\n"
        f"{freq_model.summary()}",
        encoding="utf-8",
    )

    predictions = policies.copy()
    predictions["expected_claim_count"] = freq_model.predict(
        policies, offset=np.log(policies[exposure_column])
    )
    predictions["expected_frequency"] = (
        predictions["expected_claim_count"] / policies[exposure_column]
    )
    spec = {
        "run_date": date.today().isoformat(),
        "mode": "generic",
        "period_column": period_column,
        "exposure_column": exposure_column,
        "claim_count_column": count_column,
        "categorical_features": config["features"].get("categorical", []),
        "numeric_features": config["features"].get("numeric", []),
        "frequency_family": frequency_config.get("family", "poisson"),
        "frequency_alpha": frequency_config.get("alpha"),
        "frequency_formula": freq_formula,
        "frequency_parameters": {key: float(value) for key, value in freq_model.params.items()},
        "periods": sorted(policies[period_column].astype(str).unique()),
        "record_count": int(len(policies)),
        "exposure_total": float(policies[exposure_column].sum()),
        "observed_claim_count": int(policies[count_column].sum()),
        "fitted_claim_count": float(predictions["expected_claim_count"].sum()),
        "frequency_deviance_explained": float(
            1 - freq_model.deviance / freq_model.null_deviance
        ),
    }

    if severity:
        sev_model = severity["model"]
        claims_joined = severity["claims"]
        predictions["expected_severity"] = sev_model.predict(predictions)
        predictions["indicated_loss_cost"] = (
            predictions["expected_frequency"] * predictions["expected_severity"]
        )
        spec["severity"] = {
            "family": "gamma",
            "formula": severity["formula"],
            "parameters": {key: float(value) for key, value in sev_model.params.items()},
            "basis": (
                "capped attritional claim amount"
                if severity["cap"] is not None else "uncapped claim amount"
            ),
            "claim_amount_column": severity["amount_column"],
            "cap": severity["cap"],
            "matched_claim_count": int(len(claims_joined)),
            "unmatched_claim_count": severity["unmatched_claims"],
            "common_periods": severity["common_periods"],
            "excluded_policy_periods": severity["excluded_policy_periods"],
            "excluded_claim_periods": severity["excluded_claim_periods"],
            "excluded_policy_rows": severity["excluded_policy_rows"],
            "excluded_claim_rows": severity["excluded_claim_rows"],
            "unmatched_claim_rows": severity["unmatched_claims"],
            "policy_reported_claim_count": int(policies[count_column].sum()),
            "matched_claim_rows": int(len(claims_joined)),
            "matched_claim_row_difference": int(policies[count_column].sum()) - len(claims_joined),
        }
        (output_dir / "generic_severity_summary.txt").write_text(
            f"Formula: {severity['formula']}\n"
            f"Family: gamma\nClaims: {len(claims_joined):,}\n\n{sev_model.summary()}",
            encoding="utf-8",
        )

    if severity and severity["cap"] is not None:
        predictions["indicated_attritional_loss_cost"] = (
            predictions["expected_frequency"] * predictions["expected_severity"]
        )
    elif severity:
        predictions["indicated_loss_cost"] = (
            predictions["expected_frequency"] * predictions["expected_severity"]
        )
    predictions.to_csv(output_dir / "generic_policy_predictions.csv", index=False)
    (output_dir / "generic_model_spec.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )
    print(f"Generic model fitting complete. Artifacts written to {output_dir}")


# Paths resolve relative to this workspace, so the run is reproducible from any checkout.
WORKSPACE = Path(__file__).resolve().parents[2]
DATA_DIR = WORKSPACE / "02_data_prep" / "output"
REPORT_DIR = WORKSPACE / "04_model" / "output"

# ---------------------------------------------------------------------------
# Rating structure. Band edges are right-inclusive (pd.cut). A real engagement
# would agree bands with the client; these are chosen so every class carries
# enough exposure to be credible (smallest class: DrivAge 18-21, ~4,400 years).
# ---------------------------------------------------------------------------
BANDS = {
    "DrivAge": {"edges": [17, 21, 25, 30, 40, 50, 60, 70, 200],
                "labels": ["18-21", "22-25", "26-30", "31-40", "41-50", "51-60", "61-70", "71+"]},
    "VehAge": {"edges": [-1, 0, 2, 5, 10, 15, 200],
               "labels": ["0", "1-2", "3-5", "6-10", "11-15", "16+"]},
    "VehPower": {"edges": [0, 4, 5, 6, 7, 8, 9, 11, 99],
                 "labels": ["4", "5", "6", "7", "8", "9", "10-11", "12+"]},
}
CATEGORICAL = ["Area", "VehGas"]
RATING_FACTORS = [f"{f}_band" for f in BANDS] + CATEGORICAL

# BonusMalus cap for the challenger GLM only (the production model applies the
# statutory BM/100 without re-estimating it). Above 150 the book is empty:
# 209 policies / 95 exposure-years out of 358,360.
BM_CAP = 150

# Severity is modelled on attritional claims: amounts capped at a fixed large-loss
# threshold. The excess above the cap is spread over the book as one flat loading.
# Uncapped, three claims of EUR 1.3M-4.1M (all drivers aged 18-20) drove a 5.4x
# severity relativity for the youngest band - one claim setting a tariff class.
# Threshold set by the actuary at the run-3 review (2026-09-29): EUR 50,000 fixed,
# instead of the 99th percentile (EUR 16,451) first proposed.
LARGE_LOSS_THRESHOLD = 50_000.0

# Illustrative-only loadings - NOT real pricing figures.
EXPENSE_LOADING = 0.15
PROFIT_MARGIN = 0.10


def apply_bands(df: pd.DataFrame, bands: dict = BANDS) -> pd.DataFrame:
    df = df.copy()
    for factor, b in bands.items():
        df[f"{factor}_band"] = pd.cut(df[factor], bins=b["edges"], labels=b["labels"]).astype(str)
    return df


def base_levels(policies: pd.DataFrame) -> dict:
    """Base class per factor = the class with the most exposure (standard choice:
    relativities are then read against the bulk of the book)."""
    return {f: policies.groupby(f)["Exposure"].sum().idxmax() for f in RATING_FACTORS}


def rhs(bases: dict) -> str:
    return " + ".join(f"C({f}, Treatment(reference='{bases[f]}'))" for f in RATING_FACTORS)


def bm_offset(df: pd.DataFrame) -> np.ndarray:
    return np.log(df["BonusMalus"] / 100.0)


def predict_log(params: dict, df: pd.DataFrame, spec: dict) -> np.ndarray:
    """Linear predictor from stored coefficients - the scorer 05_sensitivity and
    04_model_challenger reuse, so no stage needs the fitted statsmodels object."""
    eta = np.full(len(df), params["Intercept"])
    for f in spec["rating_factors"]:
        base = spec["base_levels"][f]
        eta = eta + df[f].map(lambda lvl: params.get(f"C({f}, Treatment(reference='{base}'))[T.{lvl}]", 0.0)).to_numpy()
    return eta


def fit_models(policies, claims, bases):
    freq_formula = f"ClaimNb ~ {rhs(bases)}"
    sev_formula = f"ClaimAmountCapped ~ {rhs(bases)}"

    # Production frequency: BonusMalus as a fixed offset (coefficient 1 by construction).
    freq = smf.glm(freq_formula, data=policies, family=sm.families.Poisson(),
                   offset=np.log(policies["Exposure"]) + bm_offset(policies)).fit()

    joined = claims.merge(policies[["IDpol", "BonusMalus"] + RATING_FACTORS], on="IDpol", how="inner")
    threshold = LARGE_LOSS_THRESHOLD
    joined["ClaimAmountCapped"] = joined["ClaimAmount"].clip(upper=threshold)
    large_loss_load = float(joined["ClaimAmount"].sum() / joined["ClaimAmountCapped"].sum())
    sev = smf.glm(sev_formula, data=joined,
                  family=sm.families.Gamma(link=sm.families.links.Log())).fit()

    # Challenger: BonusMalus slope estimated freely, capped at BM_CAP. A slope of
    # exactly 1 reproduces the statutory BM/100, so this nests the production model.
    policies = policies.assign(log_bm_c=np.log(policies["BonusMalus"].clip(upper=BM_CAP) / 100.0))
    joined = joined.assign(log_bm_c=np.log(joined["BonusMalus"].clip(upper=BM_CAP) / 100.0))
    freq_alt = smf.glm(freq_formula + " + log_bm_c", data=policies, family=sm.families.Poisson(),
                       offset=np.log(policies["Exposure"])).fit()
    sev_alt = smf.glm(sev_formula + " + log_bm_c", data=joined,
                      family=sm.families.Gamma(link=sm.families.links.Log())).fit()

    # No-BonusMalus frequency model: what the rating factors look like when
    # BonusMalus is left out entirely (the run-2 structure). Used for the
    # off-balance and confounding diagnostics only.
    freq_nobm = smf.glm(freq_formula, data=policies, family=sm.families.Poisson(),
                        offset=np.log(policies["Exposure"])).fit()

    return {"freq": freq, "sev": sev, "freq_alt": freq_alt, "sev_alt": sev_alt,
            "freq_nobm": freq_nobm, "joined": joined,
            "large_loss_threshold": threshold, "large_loss_load": large_loss_load,
            "freq_formula": freq_formula, "sev_formula": sev_formula}


def relativity_table(policies, joined, m, bases) -> pd.DataFrame:
    rows = []
    for f in RATING_FACTORS:
        expo = policies.groupby(f)["Exposure"].sum()
        nb = policies.groupby(f)["ClaimNb"].sum()
        nsev = joined.groupby(f).size()
        order = BANDS[f.replace("_band", "")]["labels"] if f.endswith("_band") else sorted(expo.index)
        for level in order:
            term = f"C({f}, Treatment(reference='{bases[f]}'))[T.{level}]"
            fr = 1.0 if level == bases[f] else float(np.exp(m["freq"].params[term]))
            sr = 1.0 if level == bases[f] else float(np.exp(m["sev"].params[term]))
            f_p = np.nan if level == bases[f] else float(m["freq"].pvalues[term])
            s_p = np.nan if level == bases[f] else float(m["sev"].pvalues[term])
            rows.append({"factor": f.replace("_band", ""), "level": level, "is_base": level == bases[f],
                         "exposure_years": round(float(expo[level]), 1), "claim_count": int(nb[level]),
                         "observed_frequency": round(float(nb[level] / expo[level]), 4),
                         "severity_claims": int(nsev.get(level, 0)),
                         "freq_relativity": round(fr, 3), "freq_p_value": f_p,
                         "sev_relativity": round(sr, 3), "sev_p_value": s_p,
                         "pure_premium_relativity": round(fr * sr, 3)})
    return pd.DataFrame(rows)


def build_example_profiles() -> pd.DataFrame:
    return pd.DataFrame([
        {"label": "Low-risk: experienced driver, low-power car, clean record",
         "VehPower": 5, "VehAge": 8, "DrivAge": 50, "BonusMalus": 50, "VehGas": "Regular", "Area": "B"},
        {"label": "Average policy (near-median values)",
         "VehPower": 6, "VehAge": 6, "DrivAge": 44, "BonusMalus": 50, "VehGas": "Regular", "Area": "C"},
        {"label": "Young driver, high-power car",
         "VehPower": 10, "VehAge": 2, "DrivAge": 22, "BonusMalus": 90, "VehGas": "Diesel", "Area": "E"},
        {"label": "High bonus-malus (poor claims history)",
         "VehPower": 7, "VehAge": 5, "DrivAge": 40, "BonusMalus": 150, "VehGas": "Regular", "Area": "D"},
        {"label": "Older vehicle, older driver",
         "VehPower": 4, "VehAge": 15, "DrivAge": 65, "BonusMalus": 50, "VehGas": "Diesel", "Area": "A"},
    ])


def technical_premium(freq, sev, df, large_loss_load):
    """Annual technical premium. Frequency is per exposure-year at the policy's own
    BonusMalus (the offset carries BM/100); severity has no BonusMalus term and is
    grossed up from attritional to all claims by the flat large-loss loading."""
    f = freq.predict(df, offset=bm_offset(df))
    s = sev.predict(df) * large_loss_load
    pure = f * s
    return f, s, pure, pure / (1 - EXPENSE_LOADING - PROFIT_MARGIN)


def main_reference():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    policies = pd.read_csv(DATA_DIR / "cleaned_policies.csv")
    claims = pd.read_csv(DATA_DIR / "cleaned_claims.csv")
    policies = apply_bands(policies[policies["Exposure"] > 0])
    bases = base_levels(policies)

    print("Fitting GLMs (production, BonusMalus challenger, no-BonusMalus diagnostic)...")
    m = fit_models(policies, claims, bases)
    joined = m["joined"]
    freq, sev = m["freq"], m["sev"]

    # ---- model summaries ----
    for name, model, formula, n, extra in [
        ("frequency_model_summary.txt", freq, m["freq_formula"], len(policies),
         "Poisson GLM, log link, offset = log(Exposure) + log(BonusMalus/100)"),
        ("severity_model_summary.txt", sev, m["sev_formula"], len(joined), "Gamma GLM, log link"),
        ("bonus_malus_challenger_summary.txt", m["freq_alt"], m["freq_formula"] + " + log(min(BM,150)/100)",
         len(policies), "Challenger frequency GLM: BonusMalus slope estimated freely, offset = log(Exposure)"),
    ]:
        with open(REPORT_DIR / name, "w", encoding="utf-8") as fh:
            fh.write(f"{extra}\nFormula: {formula}\nFitted on: {n:,} rows\nDate: {date.today().isoformat()}\n")
            fh.write(f"Deviance explained: {1 - model.deviance / model.null_deviance:.4%}\n\n")
            fh.write(str(model.summary()))

    # ---- relativities (tariff table) ----
    rel = relativity_table(policies, joined, m, bases)
    rel.to_csv(REPORT_DIR / "relativities.csv", index=False)
    base_freq = float(np.exp(freq.params["Intercept"]))
    base_sev = float(np.exp(sev.params["Intercept"]))

    # ---- portfolio balance (item: off-balance of the run-2 post-hoc multiplier) ----
    ll = m["large_loss_load"]
    sev_all = sev.predict(policies) * ll
    exp_loss_prod = float((freq.predict(policies, offset=np.log(policies["Exposure"]) + bm_offset(policies)) * sev_all).sum())
    mu_nobm = m["freq_nobm"].predict(policies, offset=np.log(policies["Exposure"]))
    exp_loss_nobm = float((mu_nobm * sev_all).sum())
    run2_style = float((mu_nobm * sev_all * policies["BonusMalus"] / 100.0).sum())
    observed_claims = int(policies["ClaimNb"].sum())
    fitted_claims = float(freq.predict(policies, offset=np.log(policies["Exposure"]) + bm_offset(policies)).sum())
    balance = pd.DataFrame([
        {"measure": "Observed claim count", "value": observed_claims},
        {"measure": "Observed claim amount, matched claims (EUR)", "value": round(float(joined["ClaimAmount"].sum()))},
        {"measure": "Fitted claim count, production GLM (BM offset)", "value": round(fitted_claims, 1)},
        {"measure": "Expected losses, production GLM (EUR)", "value": round(exp_loss_prod)},
        {"measure": "Expected losses, GLM without BonusMalus (EUR)", "value": round(exp_loss_nobm)},
        {"measure": "Run-2 method: no-BM GLM x BM/100 afterwards (EUR)", "value": round(run2_style)},
        {"measure": "Run-2 method as share of expected losses", "value": round(run2_style / exp_loss_nobm, 4)},
        {"measure": "Production as share of expected losses (no-BM GLM)", "value": round(exp_loss_prod / exp_loss_nobm, 4)},
    ])
    balance.to_csv(REPORT_DIR / "portfolio_balance.csv", index=False)

    # ---- BonusMalus challenger comparison ----
    fa, sa = m["freq_alt"], m["sev_alt"]
    lr = freq.deviance - fa.deviance  # production = challenger with slope fixed at 1
    beta, se = float(fa.params["log_bm_c"]), float(fa.bse["log_bm_c"])
    bm_levels = [50, 60, 70, 80, 90, 100, 125, 150]
    band_obs = (policies.assign(bm_band=pd.cut(policies["BonusMalus"], [0, 50, 60, 70, 80, 90, 100, 125, 150, 999]))
                .groupby("bm_band", observed=True)[["ClaimNb", "Exposure"]].sum())
    obs_freq = (band_obs["ClaimNb"] / band_obs["Exposure"]).to_numpy()
    comp = pd.DataFrame({
        "BonusMalus": bm_levels,
        "statutory_factor_BM_over_100": [b / 100 for b in bm_levels],
        "challenger_fitted_factor": [round((min(b, BM_CAP) / 100) ** beta, 3) for b in bm_levels],
        "observed_frequency_band_upper_edge": [round(x, 4) for x in obs_freq[:len(bm_levels)]],
    })
    comp.to_csv(REPORT_DIR / "bonus_malus_challenger_comparison.csv", index=False)
    bm_stats = {
        "freq_slope": beta, "freq_slope_se": se,
        "freq_slope_ci95": [beta - 1.96 * se, beta + 1.96 * se],
        "sev_slope": float(sa.params["log_bm_c"]), "sev_slope_p": float(sa.pvalues["log_bm_c"]),
        "lr_stat_slope_eq_1": float(lr), "lr_p_value": float(stats.chi2.sf(lr, 1)),
        "aic_production": float(freq.aic), "aic_challenger": float(fa.aic),
    }

    # ---- confounding: driver-age relativities with vs without BonusMalus ----
    def age_rel(model):
        out = {}
        for lvl in BANDS["DrivAge"]["labels"]:
            t = f"C(DrivAge_band, Treatment(reference='{bases['DrivAge_band']}'))[T.{lvl}]"
            out[lvl] = 1.0 if lvl == bases["DrivAge_band"] else float(np.exp(model.params[t]))
        return out
    conf = pd.DataFrame({"DrivAge_band": BANDS["DrivAge"]["labels"]})
    for col, model in [("without_bonus_malus", m["freq_nobm"]), ("bm_offset_production", freq),
                       ("bm_free_slope_challenger", fa)]:
        r = age_rel(model)
        conf[col] = [round(r[l], 3) for l in conf["DrivAge_band"]]
    mean_bm = policies.groupby("DrivAge_band").apply(
        lambda g: np.average(g["BonusMalus"], weights=g["Exposure"]), include_groups=False)
    conf["mean_bonus_malus"] = [round(float(mean_bm[l]), 1) for l in conf["DrivAge_band"]]
    conf.to_csv(REPORT_DIR / "drivage_confounding.csv", index=False)

    # ---- technical premium, example profiles ----
    examples = apply_bands(build_example_profiles())
    f, s, pure, tp = technical_premium(freq, sev, examples, ll)
    examples["predicted_freq_per_year"] = f
    examples["predicted_severity"] = s
    examples["bonus_malus_factor"] = examples["BonusMalus"] / 100.0
    examples["pure_premium"] = pure
    examples["technical_premium_illustrative"] = tp
    examples.drop(columns=[c for c in examples if c.endswith("_band")]).to_csv(
        REPORT_DIR / "technical_premium_examples.csv", index=False)

    # ---- spec + fitted models for downstream stages ----
    spec = {"run_date": date.today().isoformat(), "bands": BANDS, "categorical": CATEGORICAL,
            "rating_factors": RATING_FACTORS, "base_levels": bases,
            "freq_formula": m["freq_formula"], "sev_formula": m["sev_formula"],
            "bonus_malus": "offset log(BonusMalus/100) in the frequency GLM; not in severity",
            "base_frequency_at_bm100": base_freq, "base_severity_attritional": base_sev,
            "large_loss_threshold": m["large_loss_threshold"], "large_loss_load": ll,
            "expense_loading": EXPENSE_LOADING, "profit_margin": PROFIT_MARGIN,
            "deviance_explained": {"frequency": 1 - freq.deviance / freq.null_deviance,
                                   "severity": 1 - sev.deviance / sev.null_deviance},
            "bonus_malus_challenger": bm_stats,
            # Downstream stages score from these coefficients (predict_log) - no
            # pickled results object. The run-2 models.pkl was ~220 MB because
            # statsmodels keeps the full 678k-row design matrices.
            "freq_params": {k: float(v) for k, v in freq.params.items()},
            "sev_params": {k: float(v) for k, v in sev.params.items()}}
    (REPORT_DIR / "glm_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    stale = REPORT_DIR / "models.pkl"
    if stale.exists():
        stale.unlink()

    print("\nModel fitting complete.")
    print(f"  Frequency: {len(freq.params)} params, deviance {freq.deviance:,.1f}, "
          f"explained {spec['deviance_explained']['frequency']:.2%}")
    print(f"  Severity:  {len(sev.params)} params, deviance {sev.deviance:,.1f}, "
          f"explained {spec['deviance_explained']['severity']:.2%}")
    print(f"  Base frequency (BM=100): {base_freq:.4f}; base attritional severity EUR {base_sev:,.0f}; "
          f"large-loss threshold EUR {m['large_loss_threshold']:,.0f}, load {ll:.4f}")
    print(f"\nPortfolio balance:\n{balance.to_string(index=False)}")
    print(f"\nBonusMalus challenger: slope {beta:.3f} (95% CI {beta - 1.96 * se:.3f}-{beta + 1.96 * se:.3f}), "
          f"LR vs slope=1: {lr:.1f} (p={bm_stats['lr_p_value']:.2g}); severity slope {bm_stats['sev_slope']:.3f} "
          f"(p={bm_stats['sev_slope_p']:.2g})")
    print(f"\n{comp.to_string(index=False)}")
    print(f"\nDriver-age confounding:\n{conf.to_string(index=False)}")
    print(f"\nRelativities:\n{rel[['factor', 'level', 'exposure_years', 'freq_relativity', 'sev_relativity', 'pure_premium_relativity']].to_string(index=False)}")
    print(f"\nExample premiums (illustrative loadings):\n{examples[['label', 'predicted_freq_per_year', 'predicted_severity', 'technical_premium_illustrative']].to_string(index=False)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON configuration for generic mapped-data fitting; without it, run the reference model.",
    )
    args = parser.parse_args()
    if args.config:
        main_generic(args.config.resolve())
    else:
        main_reference()


if __name__ == "__main__":
    main()
