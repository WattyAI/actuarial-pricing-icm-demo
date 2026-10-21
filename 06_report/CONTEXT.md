---
type: model
model: frontier
inputs: [ ../02_data_prep/output/, ../03_assumptions/output/, ../04_model/output/, ../04_model_challenger/output/, ../05_sensitivity/output/ ]
output: ./output/
---
## Job
Write model documentation for the current pricing work. Explain the actual
purpose, source population and periods, selected methods, findings,
limitations, assumptions, and governance decisions. Use the report structure
below as a guide; adapt it to the applicable model and requirements rather
than treating benchmark-specific sections or outputs as universal.

## Report layout (ten sections plus header block)
1. Executive summary (written last) · 2. Scope and purpose · 3. Data and data quality · 4. Methodology · 5. Model and results · 6. Validation and challenger · 7. Sensitivity · 8. Assumptions and limitations · 9. Governance and sign-offs · 10. Appendices (assumptions register, data dictionary, glossary).

## Inputs
| Source | Path | Section/scope | Why |
|---|---|---|---|
| Data quality | `../02_data_prep/output/` | mapped sources, period overlap, exclusions, reconciliation, transformations | data and quality discussion |
| Assumptions register | `../03_assumptions/output/Assumptions_and_Judgements.md` | all | §5 and §8 |
| Model outputs | `../04_model/output/` | artifacts produced by the selected specification | model and results |
| Optional challenger outputs | `../04_model_challenger/output/` | only if a challenger is used | validation and diagnostics |
| Sensitivity | `../05_sensitivity/output/` | scenarios selected for this run | sensitivity discussion, if applicable |

## Process
1. Use the layout above where suitable; describe the actual data, periods,
   model, outputs, and applicable sections for this run.
2. Every figure is copied from an output file, never recomputed or rounded differently between sections.
3. Disclose every judgement call and every open gate (including undecided challenger signals).
4. Write the executive summary last; limitations are never omitted.
5. Markdown only; Word/PDF conversion is a later, separate step.

## Outputs
| Artifact | Location | Format |
|---|---|---|
| Model documentation | `output/` | markdown in the agreed language(s) |
| Visual companion | `output/` | optional, when useful and supported by the selected outputs |

## Done when
`output/` holds the documentation appropriate to this run, ready for
`07_check/`.
