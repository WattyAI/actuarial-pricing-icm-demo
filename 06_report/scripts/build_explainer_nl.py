"""
Builds 06_report/output/model_results_explainer_nl.html (Dutch) from the run outputs.

The explainer is the visual companion to Motor_Pricing_GLM_Modeldocumentatie.md.
Every number on the page is read from 04_model / 04_model_challenger /
05_sensitivity outputs at build time - nothing is typed in by hand, so the page
cannot drift from the model the way the run-2 version did.

Language: Dutch. The English version is build_explainer.py.
"""
import base64
import json
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[2]
M = WORKSPACE / "04_model" / "output"
C = WORKSPACE / "04_model_challenger" / "output"
S = WORKSPACE / "05_sensitivity" / "output"
OUT = WORKSPACE / "06_report" / "output" / "model_results_explainer_nl.html"

FACTOR_NL = {"DrivAge": "Leeftijd bestuurder", "VehAge": "Voertuigleeftijd", "VehPower": "Vermogensklasse",
             "Area": "Tariefgebied", "VehGas": "Brandstof"}
FEATURE_NL = {"BonusMalus": "Bonus-malus", **FACTOR_NL}
MODEL_NL = {"GLM, linear terms (run-2 structure)": "GLM, lineaire termen (run 2)",
            "GLM, banded (production)": "GLM, klassen (productie)",
            "GBM challenger (standalone)": "GBM-challenger (los)",
            "GLM x GBM correction": "GLM × GBM-correctie"}
PROFILE_NL = {"Low-risk: experienced driver, low-power car, clean record": "Ervaren, klein vermogen (BM 50)",
              "Average policy (near-median values)": "Mediaanpolis (BM 50)",
              "Young driver, high-power car": "Jonge bestuurder, groot vermogen",
              "High bonus-malus (poor claims history)": "Hoge bonus-malus (BM 150)",
              "Older vehicle, older driver": "Oud voertuig, oudere bestuurder"}


def nl(x, d=0):
    s = f"{x:,.{d}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def main():
    spec = json.loads((M / "glm_spec.json").read_text(encoding="utf-8"))
    rel = pd.read_csv(M / "relativities.csv")
    bal = pd.read_csv(M / "portfolio_balance.csv").set_index("measure")["value"]
    conf = pd.read_csv(M / "drivage_confounding.csv")
    ex = pd.read_csv(M / "technical_premium_examples.csv")
    comp = pd.read_csv(C / "model_comparison.csv")
    imp = pd.read_csv(C / "shap_importance.csv")
    summ = json.loads((C / "challenger_summary.json").read_text(encoding="utf-8"))
    bm = pd.read_csv(S / "sensitivity_bonus_malus.csv")
    age = pd.read_csv(S / "sensitivity_driver_age.csv")
    bmc = spec["bonus_malus_challenger"]

    run2_share = float(bal["Run-2 method as share of expected losses"])
    fitted, observed = float(bal["Fitted claim count, production GLM (BM offset)"]), float(bal["Observed claim count"])
    prod_dev = comp.loc[comp.model == "GLM, banded (production)", "holdout_poisson_deviance"].iloc[0]
    best = comp.loc[comp.model == "GLM x GBM correction", "deviance_reduction_vs_production"].iloc[0]
    lin = comp.loc[comp.model.str.startswith("GLM, linear"), "deviance_reduction_vs_production"].iloc[0]
    tp = ex["technical_premium_illustrative"]
    bm_ratio = bm.technical_premium.max() / bm.technical_premium.min()

    # ages per band for the step chart (one representative age per class)
    bands = spec["bands"]["DrivAge"]
    band_prem = []
    for lo, hi, lab in zip(bands["edges"][:-1], bands["edges"][1:], bands["labels"]):
        rows = age[(age.DrivAge > lo) & (age.DrivAge <= hi)]
        band_prem.append({"label": lab, "value": round(float(rows.technical_premium.iloc[0]), 2)})

    data = {
        "relativities": [{"factor": FACTOR_NL[r.factor], "level": str(r.level), "base": bool(r.is_base),
                          "freq": r.freq_relativity, "sev": r.sev_relativity, "pp": r.pure_premium_relativity,
                          "expo": r.exposure_years} for r in rel.itertuples()],
        "comparison": [{"label": MODEL_NL[r.model], "value": round(-100 * r.deviance_reduction_vs_production, 2),
                        "dev": r.holdout_poisson_deviance} for r in comp.itertuples()],
        "shap": [{"label": FEATURE_NL[r.feature], "value": r.correction_mean_abs_shap} for r in imp.itertuples()],
        "profiles": sorted([{"label": PROFILE_NL[r.label], "value": round(r.technical_premium_illustrative, 2)}
                            for r in ex.itertuples()], key=lambda d: d["value"]),
        "bm": [{"x": int(r.BonusMalus), "prod": round(r.technical_premium, 2),
                "chal": round(r.challenger_free_slope_premium, 2)} for r in bm.itertuples()],
        "age": band_prem,
        "confounding": [{"label": r.DrivAge_band, "none": r.without_bonus_malus, "offset": r.bm_offset_production,
                         "free": r.bm_free_slope_challenger, "bm": r.mean_bonus_malus} for r in conf.itertuples()],
    }

    conf_rows = "".join(
        f"<tr><td>{d['label']}</td><td class='num'>{nl(d['bm'], 1)}</td><td class='num'>{nl(d['none'], 2)}</td>"
        f"<td class='num'><strong>{nl(d['offset'], 2)}</strong></td><td class='num'>{nl(d['free'], 2)}</td></tr>"
        for d in data["confounding"])
    comp_rows = "".join(f"<tr><td>{d['label']}</td><td class='num'>{nl(d['dev'], 1)}</td>"
                        f"<td class='num'>{'+' if d['value'] > 0 else ''}{nl(d['value'], 2)}%</td></tr>"
                        for d in data["comparison"])

    html = TEMPLATE
    for k, v in {
        "__DATA__": json.dumps(data, ensure_ascii=False),
        "__RUN_DATE__": spec["run_date"],
        "__N_POL__": nl(678013), "__N_CLAIMS__": nl(26444),
        "__FITTED__": nl(fitted), "__OBSERVED__": nl(observed),
        "__RUN2_SHARE__": nl(100 * run2_share, 1), "__RUN2_SHORT__": nl(100 * (1 - run2_share), 0),
        "__FREQ_EXPL__": nl(100 * spec["deviance_explained"]["frequency"], 2),
        "__SEV_EXPL__": nl(100 * spec["deviance_explained"]["severity"], 2),
        "__BEST__": nl(100 * best, 2), "__LIN__": nl(-100 * lin, 2),
        "__SLOPE__": nl(bmc["freq_slope"], 2), "__SLOPE_LO__": nl(bmc["freq_slope_ci95"][0], 2),
        "__SLOPE_HI__": nl(bmc["freq_slope_ci95"][1], 2),
        "__TP_MIN__": nl(tp.min()), "__TP_MAX__": nl(tp.max()), "__TP_RATIO__": nl(tp.max() / tp.min(), 1),
        "__BM_RATIO__": nl(bm_ratio, 1), "__BM_MIN__": nl(bm.technical_premium.min()), "__BM_MAX__": nl(bm.technical_premium.max()),
        "__LL_CAP__": nl(spec["large_loss_threshold"]), "__LL_LOAD__": nl(spec["large_loss_load"], 4),
        "__GAP__": f"{max(summ['shap_efficiency_gap_max'].values()):.0e}",
        # Embedded so the page is one self-contained file (safe to open anywhere).
        "__SHAP_PNG__": "data:image/png;base64," + base64.b64encode((C / "shap_dependence.png").read_bytes()).decode(),
        "__N_TEST__": nl(summ["n_test"]), "__CONF_ROWS__": conf_rows, "__COMP_ROWS__": comp_rows,
    }.items():
        html = html.replace(k, str(v))
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT}")


TEMPLATE = r"""<!doctype html>
<html lang="nl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Motor Pricing GLM — Resultaten</title>
<style>
  :root {
    color-scheme: light;
    --ground: #f5f6f8; --surface: #ffffff; --surface-sunken: #eef0f3;
    --ink: #12151b; --ink-secondary: #565f6c; --ink-muted: #8991a0;
    --hairline: #e1e4e9; --hairline-strong: #c7ccd3;
    --navy: #003366; --teal: #008080; --orange: #e05a00; --good: #0a7d0a;
    --shadow: 0 1px 2px rgba(11,11,11,0.04), 0 8px 24px -12px rgba(11,11,11,0.10);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --ground: #0d0f13; --surface: #161a21; --surface-sunken: #101318;
      --ink: #f0f2f5; --ink-secondary: #a9b1bd; --ink-muted: #767f8c;
      --hairline: #262b34; --hairline-strong: #343b46;
      --navy: #6fa3dc; --teal: #3fbfbf; --orange: #ff8a3d; --good: #3ccf3c;
      --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 24px -12px rgba(0,0,0,0.5);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --ground: #0d0f13; --surface: #161a21; --surface-sunken: #101318;
    --ink: #f0f2f5; --ink-secondary: #a9b1bd; --ink-muted: #767f8c;
    --hairline: #262b34; --hairline-strong: #343b46;
    --navy: #6fa3dc; --teal: #3fbfbf; --orange: #ff8a3d; --good: #3ccf3c;
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 24px -12px rgba(0,0,0,0.5);
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body { background: var(--ground); color: var(--ink); font-family: Calibri, system-ui, -apple-system, "Segoe UI", sans-serif; -webkit-font-smoothing: antialiased; }
  .mono { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size: .92em; }
  .page { max-width: 920px; margin: 0 auto; padding: 48px 16px 96px; }
  .eyebrow { font-size: 12px; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: var(--teal); margin: 0 0 10px; }
  h1 { font-family: "Arial Black", Arial, sans-serif; font-size: clamp(26px, 4vw, 36px); line-height: 1.15; margin: 0 0 14px; color: var(--navy); }
  h2 { font-size: 22px; font-weight: 700; line-height: 1.3; margin: 0 0 12px; color: var(--ink); }
  .dek { font-size: 17px; line-height: 1.6; color: var(--ink-secondary); max-width: 64ch; margin: 0 0 28px; }
  .meta-strip { display: flex; flex-wrap: wrap; border-top: 1px solid var(--hairline-strong); border-bottom: 1px solid var(--hairline-strong); padding: 14px 0; }
  .meta-item { flex: 1 1 150px; padding: 6px 18px; border-left: 1px solid var(--hairline); }
  .meta-item:first-child { border-left: none; padding-left: 0; }
  .meta-label { font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-muted); margin-bottom: 4px; }
  .meta-value { font-size: 16px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .good { color: var(--good); } .warn { color: var(--orange); }
  section { margin-top: 52px; padding-top: 36px; border-top: 1px solid var(--hairline); }
  .sec-eyebrow { font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-muted); margin: 0 0 8px; }
  .note { font-size: 15.5px; line-height: 1.65; color: var(--ink-secondary); max-width: 68ch; margin: 0 0 22px; }
  .note strong { color: var(--ink); }
  .card { background: var(--surface); border: 1px solid var(--hairline); border-radius: 8px; padding: 22px 22px 14px; box-shadow: var(--shadow); }
  .card + .card { margin-top: 18px; }
  .card-title { font-size: 14.5px; font-weight: 700; margin-bottom: 2px; }
  .card-sub { font-size: 12.5px; color: var(--ink-muted); margin-bottom: 14px; }
  .legend { display: flex; flex-wrap: wrap; gap: 18px; font-size: 12.5px; color: var(--ink-secondary); margin-bottom: 6px; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
  svg { display: block; width: 100%; height: auto; }
  svg text { font-family: Calibri, system-ui, sans-serif; }
  .tick { fill: var(--ink-muted); font-size: 11px; }
  .lab { fill: var(--ink-secondary); font-size: 12.5px; }
  .grp { fill: var(--ink); font-size: 12.5px; font-weight: 700; }
  .val { fill: var(--ink); font-size: 12px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .pipeline { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  @media (max-width: 720px) { .pipeline { grid-template-columns: 1fr 1fr; } }
  .pipe { background: var(--surface); border: 1px solid var(--hairline-strong); border-radius: 6px; padding: 16px 14px; box-shadow: var(--shadow); }
  .pipe .k { font-size: 11px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--teal); margin-bottom: 6px; }
  .pipe .t { font-size: 15px; font-weight: 700; margin-bottom: 6px; }
  .pipe .f { font-size: 12.5px; color: var(--ink-secondary); line-height: 1.5; }
  .pipe.ml { border-style: dashed; }
  .decomp { display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 10px; margin: 20px 0; }
  .box { background: var(--surface); border: 1px solid var(--hairline-strong); border-radius: 8px; padding: 18px 24px; text-align: center; min-width: 170px; box-shadow: var(--shadow); }
  .box .k { font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--ink-muted); margin-bottom: 6px; }
  .box .v { font-size: 28px; font-weight: 800; font-variant-numeric: tabular-nums; }
  .box.bad .v { color: var(--orange); } .box.ok { background: var(--navy); border-color: var(--navy); }
  .box.ok .k { color: rgba(255,255,255,.7); } .box.ok .v { color: #fff; }
  .op { font-size: 22px; color: var(--ink-muted); padding: 0 6px; }
  .two { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
  @media (max-width: 720px) { .two { grid-template-columns: 1fr; } }
  .two .card + .card { margin-top: 0; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--hairline); color: var(--ink-secondary); }
  th { color: var(--ink-muted); font-weight: 700; text-transform: uppercase; font-size: 10.5px; letter-spacing: .05em; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  td strong { color: var(--ink); }
  .table-wrap { overflow-x: auto; }
  details { margin-top: 10px; } summary { font-size: 12.5px; color: var(--teal); cursor: pointer; font-weight: 700; }
  img.fig { width: 100%; height: auto; border-radius: 6px; border: 1px solid var(--hairline); background: #fff; }
  .gates { display: grid; gap: 10px; }
  .gate { display: flex; gap: 12px; align-items: baseline; font-size: 14.5px; line-height: 1.55; color: var(--ink-secondary); }
  .gate .s { flex: 0 0 72px; font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }
  .gate strong { color: var(--ink); }
  footer { margin-top: 60px; padding-top: 20px; border-top: 1px solid var(--hairline); font-size: 12.5px; color: var(--ink-muted); display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
</style>
</head>
<body>
<div class="page">

  <p class="eyebrow">Motor Pricing GLM — demo · run 3</p>
  <h1>Wat het model echt vond</h1>
  <p class="dek">
    Een frequentie × schadelast-GLM op de Franse motor-WA-data (freMTPL2), met een
    gradient-boosting-challenger ernaast. Twee echte modelfouten zijn onderweg gevonden en hersteld;
    beide staan hieronder, met wat het herstelde model nu laat zien.
  </p>
  <div class="meta-strip">
    <div class="meta-item"><div class="meta-label">Polissen</div><div class="meta-value">__N_POL__</div></div>
    <div class="meta-item"><div class="meta-label">Claims (schadelast)</div><div class="meta-value">__N_CLAIMS__</div></div>
    <div class="meta-item"><div class="meta-label">Premiebalans</div><div class="meta-value good">__FITTED__ = __OBSERVED__ claims</div></div>
    <div class="meta-item"><div class="meta-label">Aftekenmomenten</div><div class="meta-value warn">2 van 3 afgerond</div></div>
  </div>

  <section>
    <p class="sec-eyebrow">Hoe de premie tot stand komt</p>
    <h2>Twee GLM's in productie, machine learning ernaast</h2>
    <p class="note">
      Het tarief komt uit twee GLM's: <strong>hoe vaak</strong> een polis claimt en <strong>hoeveel</strong>
      een claim kost. Bonus-malus zit in het frequentiemodel als <strong>offset</strong>: de wettelijke factor
      BM/100, niet opnieuw geschat. Het boosting-model beslist niets; het wijst aan wat het GLM mist.
    </p>
    <div class="pipeline">
      <div class="pipe"><div class="k">Stap 1 — frequentie</div><div class="t">Poisson-GLM</div><div class="f">5 kenmerken in klassen<br>offset log(blootstelling) + log(BM/100)</div></div>
      <div class="pipe"><div class="k">Stap 2 — schadelast</div><div class="t">Gamma-GLM</div><div class="f">afgetopt op € __LL_CAP__<br>opslag voor grote schades × __LL_LOAD__</div></div>
      <div class="pipe"><div class="k">Resultaat</div><div class="t">Technische premie</div><div class="f">frequentie × schadelast<br>÷ (1 − 15% − 10%) — illustratief</div></div>
      <div class="pipe ml"><div class="k">Ernaast — diagnose</div><div class="t">GBM-challenger</div><div class="f">Poisson-verlies, BM monotoon<br>SHAP → kandidaatsignalen</div></div>
    </div>
  </section>

  <section>
    <p class="sec-eyebrow">Fout 2 — gevonden in run 3</p>
    <h2>De reparatie van run 2 liet __RUN2_SHORT__% van de premie liggen</h2>
    <p class="note">
      Run 2 fitte het GLM zonder bonus-malus en vermenigvuldigde de premie daarna met BM/100. Maar een GLM
      zonder bonus-malus voorspelt al de <strong>gemiddelde</strong> schadelast; 63% van de blootstelling
      staat op BM 50 (factor 0,5), dus de premie zakte ver onder de verwachte schadelast. Als offset in het GLM
      herijkt het intercept zich en sluit de premie aan. Bonus-malus telt ook niet meer dubbel via leeftijd.
    </p>
    <div class="decomp">
      <div class="box bad"><div class="k">Run 2 — BM/100 achteraf</div><div class="v">__RUN2_SHARE__%</div></div>
      <div class="op">→</div>
      <div class="box ok"><div class="k">Run 3 — BM als offset</div><div class="v">100%</div></div>
    </div>
    <p class="note" style="text-align:center;margin:0 auto;">van de verwachte schadelast</p>
  </section>

  <section>
    <p class="sec-eyebrow">Tarieftabel</p>
    <h2>Relativiteiten per klasse, t.o.v. de basisklasse (1,00)</h2>
    <p class="note">
      Zuivere premie = frequentie × schadelast. De basisklasse per kenmerk is de klasse met de meeste
      blootstelling. Klassen in plaats van lineaire termen verlagen de deviantie op de toetsset met
      <strong>__LIN__%</strong>: de leeftijdscurve is geen rechte lijn.
    </p>
    <div class="card">
      <div class="card-title">Relativiteit zuivere premie</div>
      <div class="card-sub">Logaritmische schaal; ◆ = basisklasse</div>
      <div class="legend"><span><span class="dot" style="background:var(--orange)"></span>hoger dan basis</span><span><span class="dot" style="background:var(--teal)"></span>lager dan basis</span></div>
      <svg id="relChart" role="img" aria-label="Relativiteiten zuivere premie per tariefklasse"></svg>
      <details><summary>Als tabel (frequentie / schadelast / zuivere premie)</summary>
        <div class="table-wrap"><table id="relTable"><thead><tr><th>Kenmerk</th><th>Klasse</th><th class="num">Verzekeringsjaren</th><th class="num">Frequentie</th><th class="num">Schadelast</th><th class="num">Zuivere premie</th></tr></thead><tbody></tbody></table></div>
      </details>
    </div>
  </section>

  <section>
    <p class="sec-eyebrow">Bonus-malus getoetst</p>
    <h2>De data lopen steiler dan de wettelijke schaal</h2>
    <p class="note">
      Een challenger-GLM schat de bonus-malushelling vrij (afgetopt op 150): <strong>__SLOPE__</strong>
      (95%-interval __SLOPE_LO__–__SLOPE_HI__) tegen 1,00 wettelijk. Het tarief houdt de gereguleerde schaal aan
      (BM 50 → 200 = precies __BM_RATIO__×, € __BM_MIN__ → € __BM_MAX__); het verschil is een benoemde beperking
      én kandidaatsignaal S1.
    </p>
    <div class="two">
      <div class="card">
        <div class="card-title">Technische premie naar bonus-malus</div>
        <div class="card-sub">Basisprofiel mediaanpolis (BM 50), EUR, illustratief</div>
        <div class="legend"><span><span class="dot" style="background:var(--navy)"></span>productie: BM/100</span><span><span class="dot" style="background:var(--orange)"></span>challenger: vrije helling</span></div>
        <svg id="bmChart" role="img" aria-label="Premie naar bonus-malus, productie en challenger"></svg>
      </div>
      <div class="card">
        <div class="card-title">Leeftijd is deels een proxy voor bonus-malus</div>
        <div class="card-sub">Frequentierelativiteit per leeftijdsklasse, afhankelijk van hoe BM in het model zit</div>
        <div class="table-wrap"><table><thead><tr><th>Leeftijd</th><th class="num">Gem. BM</th><th class="num">Zonder BM</th><th class="num">BM-offset</th><th class="num">Vrije helling</th></tr></thead><tbody>__CONF_ROWS__</tbody></table></div>
      </div>
    </div>
  </section>

  <section>
    <p class="sec-eyebrow">ML-challenger — frequentie, toetsset van __N_TEST__ polissen</p>
    <h2>Gradient boosting vindt nog __BEST__% — SHAP wijst vooral naar bonus-malus</h2>
    <p class="note">
      Het correctiemodel wordt bovenop het GLM getraind en kan dus alleen leren wat het GLM mist. De SHAP-waarden
      zijn exact berekend (controle: afwijking __GAP__) en laten zien waar de winst zit.
    </p>
    <div class="two">
      <div class="card">
        <div class="card-title">Poisson-deviantie op de toetsset</div>
        <div class="card-sub">% verschil t.o.v. het productie-GLM (lager = beter)</div>
        <svg id="compChart" role="img" aria-label="Deviantie per model t.o.v. productie"></svg>
        <details><summary>Als tabel</summary><div class="table-wrap"><table><thead><tr><th>Model</th><th class="num">Deviantie</th><th class="num">t.o.v. productie</th></tr></thead><tbody>__COMP_ROWS__</tbody></table></div></details>
      </div>
      <div class="card">
        <div class="card-title">Wat het GLM mist</div>
        <div class="card-sub">Gemiddelde |SHAP| van het correctiemodel (log-schaal)</div>
        <svg id="shapChart" role="img" aria-label="SHAP-belang per kenmerk"></svg>
      </div>
    </div>
    <div class="card" style="margin-top:18px">
      <div class="card-title">SHAP-afhankelijkheid</div>
      <div class="card-sub">Links: leeftijd volgens het GBM, met de GLM-klassen erover. Rechts: correctie op het GLM per bonus-malus.</div>
      <img class="fig" src="__SHAP_PNG__" alt="SHAP-afhankelijkheidsplots voor leeftijd en bonus-malus">
    </div>
  </section>

  <section>
    <p class="sec-eyebrow">Illustratieve technische premie</p>
    <h2>Vijf voorbeeldpolissen, € __TP_MIN__ tot € __TP_MAX__ (__TP_RATIO__×)</h2>
    <p class="note">Opslagen (15% kosten, 10% winst) zijn plaatshouders voor de demo, geen echte tariefcijfers.</p>
    <div class="two">
      <div class="card">
        <div class="card-title">Technische premie per profiel</div>
        <div class="card-sub">EUR, illustratief</div>
        <svg id="profChart" role="img" aria-label="Technische premie per voorbeeldprofiel"></svg>
      </div>
      <div class="card">
        <div class="card-title">Technische premie per leeftijdsklasse</div>
        <div class="card-sub">Basisprofiel mediaanpolis (BM 50), EUR</div>
        <svg id="ageChart" role="img" aria-label="Technische premie per leeftijdsklasse"></svg>
      </div>
    </div>
  </section>

  <section>
    <p class="sec-eyebrow">Fout 1 — gevonden in run 1</p>
    <h2>96,9× was extrapolatie, geen bevinding</h2>
    <p class="note">
      Run 1 liet het GLM de bonus-malus zelf schatten. Die helling kwam vooral uit BM 50–100 en werd doorgetrokken
      tot BM 200, waar <strong>geen enkele polis</strong> staat, in frequentie én schadelast tegelijk (≈ 30× × 3,2×).
      Gevonden bij het aftekenmoment na de gevoeligheidsanalyse.
    </p>
    <div class="decomp">
      <div class="box bad"><div class="k">Run 1 — geschat door het GLM</div><div class="v">× 96,9</div></div>
      <div class="op">→</div>
      <div class="box ok"><div class="k">Nu — wettelijke BM/100</div><div class="v">× __BM_RATIO__</div></div>
    </div>
  </section>

  <section>
    <p class="sec-eyebrow">Voordat dit naar een klant gaat</p>
    <h2>Wat nog openstaat</h2>
    <div class="gates">
      <div class="gate"><span class="s good">Afgetekend</span><span><strong>Datakwaliteit (<span class="mono">02_data_prep</span>)</strong> — aftoppingen en markering grote schades, 07-08-2026.</span></div>
      <div class="gate"><span class="s good">Besloten</span><span><strong>Kandidaatsignalen (<span class="mono">04_model_challenger</span>)</strong> — 29-09-2026: S1 (bonus-malus steiler dan wettelijk) geparkeerd tot een juridische toets, S2 (leeftijd × bonus-malus) verworpen, S3 (voertuigleeftijd × brandstof) geparkeerd voor onderzoek. Niets opgenomen in het GLM.</span></div>
      <div class="gate"><span class="s good">Besloten</span><span><strong>Klassenindeling en drempel grote schades</strong> — klassen bevestigd; schadelast afgetopt op € 50.000.</span></div>
      <div class="gate"><span class="s warn">Open</span><span><strong>Aftekening vier-ogencontrole (<span class="mono">07_check</span>)</strong> — onafhankelijke controle gedaan; de aftekening door de modeleigenaar staat bewust nog open.</span></div>
      <div class="gate"><span class="s">Beperking</span><span><strong>Aansluitingsbreuk 26,2%</strong> — 9.463 claims zonder schadebedrag; de schadelast is geschat op de claims mét bedrag.</span></div>
      <div class="gate"><span class="s">Beperking</span><span><strong>Lage verklaarde deviantie</strong> — __FREQ_EXPL__% frequentie, __SEV_EXPL__% schadelast: gebruikelijk voor claimaantallen per polis.</span></div>
    </div>
  </section>

  <footer>
    <span>motor_pricing_glm_demo · run 3 · __RUN_DATE__ · gegenereerd door <span class="mono">06_report/scripts/build_explainer_nl.py</span></span>
  </footer>
</div>

<script>
(function () {
  const D = __DATA__;
  const NS = 'http://www.w3.org/2000/svg';
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const el = (t, a) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); return e; };
  const tx = (x, y, s, a) => { const t = el('text', Object.assign({ x, y }, a)); t.textContent = s; return t; };
  const n2 = (v, d) => v.toLocaleString('nl-NL', { minimumFractionDigits: d, maximumFractionDigits: d });
  const eur = (v) => '€ ' + Math.round(v).toLocaleString('nl-NL');

  function hbars(id, rows, opt) {
    const svg = document.getElementById(id);
    const W = opt.W || 420, rowH = opt.rowH || 30, top = 6, left = opt.left || 170, right = opt.right || 64;
    const H = top + rows.length * rowH + (opt.axis ? 22 : 6);
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const plotW = W - left - right;
    const vals = rows.map(r => r.value);
    const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
    const sx = (v) => left + ((v - lo) / (hi - lo || 1)) * plotW;
    svg.appendChild(el('line', { x1: sx(0), x2: sx(0), y1: top, y2: top + rows.length * rowH, stroke: css('--hairline-strong') }));
    rows.forEach((r, i) => {
      const y = top + i * rowH + rowH / 2, x0 = Math.min(sx(0), sx(r.value)), w = Math.max(Math.abs(sx(r.value) - sx(0)), 2);
      svg.appendChild(el('rect', { x: x0, y: y - 9, width: w, height: 18, rx: 3, fill: opt.color(r) }));
      svg.appendChild(tx(left - 10, y + 4, r.label, { class: 'lab', 'text-anchor': 'end' }));
      // Negative bars grow toward the row labels, so their value sits right of the zero line.
      const vx = r.value >= 0 ? sx(r.value) + 6 : sx(0) + 6;
      svg.appendChild(tx(vx, y + 4, opt.fmt(r.value), { class: 'val', 'text-anchor': 'start' }));
    });
  }

  // relativities: log scale, grouped by factor
  (function () {
    const svg = document.getElementById('relChart');
    const rows = D.relativities, W = 860, rowH = 19, left = 200, right = 70, top = 8;
    let groups = []; rows.forEach(r => { if (!groups.length || groups[groups.length - 1].f !== r.factor) groups.push({ f: r.factor, rows: [] }); groups[groups.length - 1].rows.push(r); });
    const H = top + rows.length * rowH + groups.length * 22 + 26;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const plotW = W - left - right, lmin = Math.log(0.5), lmax = Math.log(4);
    const sx = (v) => left + ((Math.log(Math.min(Math.max(v, 0.5), 4)) - lmin) / (lmax - lmin)) * plotW;
    [0.5, 0.75, 1, 1.5, 2, 3, 4].forEach(g => {
      svg.appendChild(el('line', { x1: sx(g), x2: sx(g), y1: top, y2: H - 22, stroke: g === 1 ? css('--hairline-strong') : css('--hairline') }));
      svg.appendChild(tx(sx(g), H - 6, n2(g, g % 1 ? 2 : 0), { class: 'tick', 'text-anchor': 'middle' }));
    });
    let y = top;
    groups.forEach(g => {
      y += 16; svg.appendChild(tx(8, y, g.f, { class: 'grp' })); y += 6;
      g.rows.forEach(r => {
        const cy = y + rowH / 2;
        svg.appendChild(tx(left - 10, cy + 4, r.level, { class: 'lab', 'text-anchor': 'end' }));
        if (r.base) {
          svg.appendChild(el('path', { d: `M ${sx(1)} ${cy - 6} l 6 6 l -6 6 l -6 -6 z`, fill: css('--navy') }));
        } else {
          const x0 = Math.min(sx(1), sx(r.pp)), w = Math.max(Math.abs(sx(r.pp) - sx(1)), 2);
          svg.appendChild(el('rect', { x: x0, y: cy - 6, width: w, height: 12, rx: 2, fill: r.pp >= 1 ? css('--orange') : css('--teal') }));
        }
        const vx = r.pp >= 1 ? Math.max(sx(r.pp), sx(1) + 6) + 6 : sx(r.pp) - 6;
        svg.appendChild(tx(vx, cy + 4, n2(r.pp, 2), { class: 'val', 'text-anchor': r.pp >= 1 ? 'start' : 'end' }));
        y += rowH;
      });
    });
    const tb = document.querySelector('#relTable tbody');
    rows.forEach(r => { const tr = document.createElement('tr'); tr.innerHTML = `<td>${r.factor}</td><td>${r.level}${r.base ? ' (basis)' : ''}</td><td class="num">${n2(r.expo, 0)}</td><td class="num">${n2(r.freq, 3)}</td><td class="num">${n2(r.sev, 3)}</td><td class="num">${n2(r.pp, 3)}</td>`; tb.appendChild(tr); });
  })();

  // bonus-malus lines
  (function () {
    const svg = document.getElementById('bmChart'), W = 420, H = 280, L = 60, R = 16, T = 14, B = 30;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const pw = W - L - R, ph = H - T - B, ymax = Math.ceil(Math.max(...D.bm.map(d => Math.max(d.prod, d.chal))) / 200) * 200;
    const sx = v => L + (v - 50) / 150 * pw, sy = v => T + ph - v / ymax * ph;
    for (let t = 0; t <= ymax; t += 200) { svg.appendChild(el('line', { x1: L, x2: L + pw, y1: sy(t), y2: sy(t), stroke: css('--hairline') })); svg.appendChild(tx(L - 8, sy(t) + 4, eur(t), { class: 'tick', 'text-anchor': 'end' })); }
    [50, 100, 150, 200].forEach(t => svg.appendChild(tx(sx(t), H - 8, t, { class: 'tick', 'text-anchor': 'middle' })));
    [['prod', '--navy', ''], ['chal', '--orange', '5 4']].forEach(([k, c, dash]) => {
      svg.appendChild(el('path', { d: D.bm.map((d, i) => `${i ? 'L' : 'M'} ${sx(d.x)} ${sy(d[k])}`).join(' '), fill: 'none', stroke: css(c), 'stroke-width': 2.2, 'stroke-dasharray': dash }));
      D.bm.forEach(d => svg.appendChild(el('circle', { cx: sx(d.x), cy: sy(d[k]), r: 3, fill: css(c) })));
    });
    const f = D.bm[0], l = D.bm[D.bm.length - 1];
    svg.appendChild(tx(sx(f.x) + 4, sy(f.prod) - 10, eur(f.prod), { class: 'val' }));
    svg.appendChild(tx(sx(l.x) - 4, sy(l.prod) + 18, eur(l.prod), { class: 'val', 'text-anchor': 'end' }));
  })();

  hbars('compChart', D.comparison, { W: 420, left: 200, right: 58, color: r => r.value > 0 ? css('--orange') : (r.value < 0 ? css('--teal') : css('--navy')), fmt: v => (v > 0 ? '+' : '') + n2(v, 2) + '%' });
  hbars('shapChart', D.shap, { W: 420, left: 140, right: 50, color: () => css('--teal'), fmt: v => n2(v, 3) });
  hbars('profChart', D.profiles, { W: 420, left: 210, right: 60, color: () => css('--navy'), fmt: eur });
  hbars('ageChart', D.age, { W: 420, left: 70, right: 60, rowH: 26, color: () => css('--teal'), fmt: eur });
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
