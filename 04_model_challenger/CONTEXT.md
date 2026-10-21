---
type: script
inputs: [ ../02_data_prep/output/, ../04_model/output/ ]
output: ./output/
---
## Job
When justified by the objective and data, run a challenger or alternative
model next to the approved production model as a diagnostic. It provides
evidence for actuarial review and never changes or replaces the production
model by itself. The method and diagnostics are selected for the current run;
gradient boosting and SHAP are reference-run choices, not requirements.

## Inputs
| Source | Path | Section/scope | Why |
|---|---|---|---|
| Prepared modelling data | `../02_data_prep/output/` | approved fields and aligned periods | consistent population and predictors |
| Production model specification | `../04_model/output/` | current approved model details | comparison on an appropriate validation basis |
| Optional model-specific artifacts | relevant stage output | only where needed for the selected challenger | align diagnostics with the production specification |

## Process
Select an appropriate challenger, comparison design, validation measure, and
explanation method from the data and objective. Ensure the comparison uses
compatible populations and an independent or otherwise justified validation
basis. Consider exposure and time/policy dependencies in data splitting.
Interpret candidate signals with suitable credibility evidence, and disclose
limitations. Do not require particular algorithms, factors, monotonic
constraints, or explanation techniques.

The current `scripts/gbm_challenger.py` implements the French motor frequency
reference example and is not a generic challenger framework.

## Outputs
| Artifact | Location | Format |
|---|---|---|
| Comparison results | `output/` | suitable tabular or text format |
| Candidate evidence and diagnostics | `output/` | format appropriate to selected analysis |
| Explanation artifacts | `output/` | optional, using a suitable method where applicable |
| Run metadata and validation checks | `output/` | machine-readable or text format |

## ⏸ Human-in-the-loop gate
Before any challenger evidence changes the production specification, the
actuary reviews the evidence and records a decision and rationale in `STATE.md`.
Any promoted change is entered into `03_assumptions` and approved through the
normal model gate. The script never promotes a signal itself.

## Done when
`output/` holds the artifacts selected for this run and `STATE.md` records the
actuary's decision on each material candidate or alternative.
