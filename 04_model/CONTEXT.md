---
type: script
inputs: [ ../02_data_prep/output/, ../03_assumptions/output/ ]
output: ./output/
---
## Job
Fit the approved pricing model and calculate its indicated price. The model
structure and any required components depend on the pricing objective,
available data, and approved assumptions. **Calculations are performed by
scripts, never by the language model.**

## Inputs
| Source | Path | Section/scope | Why |
|---|---|---|---|
| Prepared exposure and claim-count data | `../02_data_prep/output/` | approved population, periods, mapped exposure and claim-count fields | frequency model |
| Optional prepared claim-level/supplemental data | `../02_data_prep/output/` | only if selected and approved for this run | severity or other approved model components |
| Assumptions register | `../03_assumptions/output/Assumptions_and_Judgements.md` | approved model choices and rationale | every modelling choice is taken from it |

## Process
The fitting script implements the model specified for the current run:
1. Use the exposure and claim-count fields mapped and approved for the run.
   Select a response distribution, link, exposure treatment, and other
   frequency-model structure through the assumptions register.
2. Include severity or other components only when the required data are
   available, aligned, and approved. Document any caps, thresholds, class
   definitions, reference levels, and loadings as run-specific assumptions.
3. A pre-calibrated external factor is optional. Use it as an offset only when
   the external basis, field mapping, direction, scale, and approval are
   documented.
4. If a challenger fit or alternative specification is included, it tests a
   documented choice and does not replace the production model without the
   actuary's explicit decision.
5. Construct the indicated premium only from components supported by the
   selected model and approved assumptions. Distinguish loss-cost indication
   from any commercial loadings.

Use `scripts/pricing_glm.py --config 04_model/model_config.json` for mapped
inputs. Start from `model_config.example.json`, set paths relative to the
workspace root, and provide column mappings and selected predictors. The
generic mode fits a Poisson or configured negative-binomial frequency GLM and
optionally a Gamma severity GLM when aligned claim-level data are supplied.
The generic mode validates mapped fields and requires claim data to match
policies by policy ID and period; it filters joined modelling data to common
periods and records unmatched claims and claim-count differences.
Frequency families supported in generic mode are Poisson and negative
binomial; the latter requires an explicitly configured positive `alpha`.
Severity is optional and currently supports Gamma with positive claim amounts.
Any configured severity cap produces an attritional severity indication and
does not by itself account for losses above the cap.

The no-argument invocation remains a French motor reference implementation.
Its output schemas and model choices are specific to that dataset. This
structural release contains no generated model artifacts.

## Outputs
| Artifact | Location | Format |
|---|---|---|
| Model specification and fitted parameters | `output/` | machine-readable artifact, as appropriate |
| Model fit, validation, and portfolio reconciliation | `output/` | tabular/text artifacts, as appropriate |
| Factor-level indications and pricing outputs | `output/` | artifacts for selected model components |

## ⏸ Gated
**Does not run until the `02_data_prep` human-in-the-loop gate has been signed off** (see that stage's `CONTEXT.md`). Running it before the actuary has reviewed the data-quality findings would defeat the point of the gate.

## Done when
`output/` holds the artifacts required by the approved model specification,
key checks are reported, and `STATE.md` records the stage status and any
applicable human decision.
