---
type: script
inputs: [ ../04_model/output/ ]
output: ./output/
---
## Job
Stress the fitted model across material, approved assumptions and input
dimensions. Select scenarios and ranges for the current pricing objective and
data. Nothing is refitted unless explicitly specified and approved.

## Inputs
| Source | Path | Section/scope | Why |
|---|---|---|---|
| Model specification and fitted parameters | `../04_model/output/` | model components and approved assumptions relevant to scenarios | score the selected outcomes without refitting |

## Process
Select factors and assumption ranges from the approved register and observed
data support. Ask the actuary to confirm material sensitivities relevant to
the intended use; do not hard-code bonus-malus, driver age, or benchmark
ranges as universal scenarios. Clearly distinguish supported values from
extrapolation.

The current `scripts/sensitivity.py` implements the French motor reference
scenarios and is not made generic by this context change.

## Outputs
| Artifact | Location | Format |
|---|---|---|
| Scenario results | `output/` | machine-readable table(s) for selected scenarios |
| Visual summaries | `output/` | only where useful for the selected scenarios |

## ⏸ Human-in-the-loop gate
The actuary reviews the scenario results before `06_report/`. Implausible
swings or unsupported extrapolation are findings to investigate, not results
to accept silently.

## Done when
`output/` holds the selected scenario artifacts and `STATE.md` records the
actuary's review and decisions.
