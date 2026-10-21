---
type: hybrid
model: local
inputs: [ ../01_intake/input/, ../01_intake/output/ ]
output: ./output/
---
## Job
Clean, validate, and structure the supplied data for modelling. The standard
frequency input is policy/exposure-unit data with an observation period,
exposure, and number of claims. Optional claim-level and other sources are used
only when relevant to the pricing objective and confirmed by the actuary.

## Inputs
| Source | Path | Section/scope | Why |
|---|---|---|---|
| Mapped exposure and claim-count data | `../01_intake/input/` | records and fields mapped during intake | frequency modelling basis |
| Optional claim-level or supplemental data | `../01_intake/input/` | only sources approved for this run | severity/loss-cost modelling or other approved analysis |
| Source provenance and mappings | `../01_intake/input/sources.md` and intake record | source, schema/field mappings, periods, keys, units | reproducibility and intake inventory |

## Outputs
| Artifact | Location | Format |
|---|---|---|
| Prepared model population(s) | `output/` | tabular data, according to the selected components |
| Data-quality report | `output/` | markdown |
| Transformation log | `output/` | markdown |

## Steps
For each run, map the supplied fields to standard concepts and validate those
mappings before cleaning. Determine the years or coverage periods represented
by each source. When sources are joined, establish their common periods and
appropriate policy-period keys; use only consistently aligned overlapping
periods in the joined modelling population. Report source totals, overlapping
and excluded periods, unmatched keys, duplicate keys, and reconciliation
differences. An inner join alone is not evidence that the sources reconcile.

Validate structure, completeness, consistency, and anomalies; standardize only
with documented rationale. Log every correction, cap, exclusion, and
transformation. Do not apply benchmark-specific thresholds automatically.
The current `scripts/clean_data.py` still implements the reference dataset's
cleaning choices; it is not schema-generic. Prepare and map a new dataset
according to this contract before using the generic mode of
`04_model/scripts/pricing_glm.py`.

## ⏸ Human-in-the-loop gate
**Before `03_assumptions/` proceeds:** The actuary reviews the run's data-quality report, including period overlap, exclusions, mappings, reconciliation, and every proposed transformation. The actuary confirms the modelling population and unresolved limitations; no benchmark-specific adjustment or threshold is presumed acceptable for a new run.

## Done when
`output/` holds the prepared modelling population and its transformation log and data-quality report; source files were not modified in place; the actuary's sign-off on the gate above is recorded in `STATE.md`.
