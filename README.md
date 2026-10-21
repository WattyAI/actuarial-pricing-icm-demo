# Actuarial Pricing Workflow — an ICM structure

An open-source demonstration of a non-life actuarial pricing workflow
organized with the **Interpretable Context Methodology (ICM)** by Jake Van
Clief and David McDermott ([paper](https://arxiv.org/abs/2603.16021),
[reference repository](https://github.com/RinDig/Interpretable-Context-Methodology)).
Each stage has its own folder and `CONTEXT.md` contract describing its purpose,
inputs, process, outputs, and completion criteria.

This public release provides the workflow structure and reusable guidance. It
does not include run data, generated model results, reports, or review
findings. Output folders are intentionally empty except for structural
documentation. Populate them only with artifacts for an authorized run, and
review data and results for sensitivity before sharing.

## Generic input approach

The standard starting point for frequency pricing is policy- or
exposure-unit data with an observation period, exposure, and number of claims.
Map supplied column names and files during intake. Claim amounts and other
sources are optional: decide with the actuary whether they are needed for the
pricing objective. When combining sources, agree the keys and common periods,
and document excluded periods, unmatched records, and reconciliation
differences.

The included scripts illustrate parts of the workflow; they are not a
schema-generic end-to-end implementation. In particular, the data-preparation
script contains reference-schema-specific choices. Review each stage contract
and adapt and validate code for the data and assumptions of an authorized run.

## Workflow layout

Stages run in order:
`01_intake` → `02_data_prep` → `03_assumptions` → `04_model` →
`04_model_challenger` → `05_sensitivity` → `06_report` → `07_check` →
`08_client_review` → `99_close`.

```text
AGENTS.md                  workflow identity and agent guidance
STATE.md                   blank run-state and sign-off template
NN_stage/CONTEXT.md        stage-specific contract
NN_stage/input/            stage inputs
NN_stage/scripts/          scripts used by a stage
NN_stage/output/           run artifacts; empty in this release
references/                workflow references
```

Human approval gates are part of the process. Record decisions in `STATE.md`;
do not proceed past a gate until the designated human has reviewed and
approved it. Keep generated outputs separate from this structural release.

## Credits

Method: Jake Van Clief and David McDermott (ICM). Modelling and documentation:
Michael Watson, with an AI assistant proposing and drafting under the gates
described above.
