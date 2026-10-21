# AGENTS.md — Actuarial Pricing ICM Workflow

This file is read by any AI coding agent (Claude Code, Codex, Cursor, …) that opens this repository. It is the routing layer of the workspace: what the project is, how the stages fit together, and the rules every agent must follow. Human readers should start with `README.md`.

## Job
A reusable, end-to-end **non-life actuarial pricing workflow** built on the Interpretable Context Methodology (ICM). Its standard frequency input is policy/exposure-unit data containing an observation period, exposure, and number of claims. Claim-level amounts and other data are optional and should be considered against the pricing objective. The repository contains a French Motor Third-Party Liability benchmark implementation; its data, variables, models, choices, and findings are an example run, not requirements or universal conclusions.

The point is the *process*: every stage is a folder, every hand-off is a file, every judgement call is recorded, and a human actuary signs off at defined gates.

## ICM layout
Method: Jake Van Clief and David McDermott, [*Interpretable Context Methodology: Folder Structure as Agentic Architecture*](https://arxiv.org/abs/2603.16021) (arXiv:2603.16021). This repository applies it to actuarial work.

| Layer | Where | What it holds |
|---|---|---|
| Identity and routing | `AGENTS.md` (this file), `README.md`, `STATE.md` | what the project is, rules, where the run stands |
| Stage contracts | `NN_stage/CONTEXT.md` | Job / Inputs / Process / Outputs / Done when, for that stage only |
| Reference material | `references/` | sources the stages cite |
| Working artifacts | `NN_stage/input/`, `scripts/`, `output/` | data, code and results; one stage's `output/` is the next stage's input |

Stages run in order:
`01_intake` → `02_data_prep` → `03_assumptions` → `04_model` → `04_model_challenger` → `05_sensitivity` → `06_report` → `07_check` → `08_client_review` → `99_close`

When working on a stage, read only that stage's `CONTEXT.md` and the inputs it lists. Do not read ahead.

## Rules for agents
1. **Calculations are done by scripts, never by the language model.** Anything numeric (cleaning, fitting, premiums, sensitivities, SHAP) lives in a `scripts/` file that can be re-run. The model writes prose around script output; it never computes or retypes figures from memory.
2. **Every figure in prose is copied from an output file.** Do not round differently between sections or recompute in the report.
3. **Nothing is fixed silently.** Every correction, cap, exclusion or assumption is logged (`02_data_prep/output/data_transformation_log.md`, `03_assumptions/output/Assumptions_and_Judgements.md`). Ambiguous cases go back to the actuary; they are not resolved unilaterally.
4. **Human-in-the-loop gates are real.** Stages marked `⏸` stop and wait for the actuary's decision, recorded in `STATE.md`. Do not run a gated stage's successor before the gate is recorded. An agent never signs a gate on the actuary's behalf.
5. **The challenger never changes the production model.** `04_model_challenger` produces evidence; the actuary decides per signal to promote, reject or park.
6. **Four-eye independence.** The check in `07_check` should be run by a reviewer that did not draft the report (a different person, or a fresh agent with no drafting context). A same-session mechanical pass is not a sign-off.
7. **Disclose limits.** Reconciliation gaps, weak fit and open decisions must appear in the report; never omit them to look cleaner.
8. **Keep stages self-contained.** Scripts resolve paths relative to the workspace root (`Path(__file__).resolve().parents[2]`). No absolute or user-specific paths.

## Data and period alignment
Intake maps supplied files and fields to standard concepts; it does not assume benchmark-specific filenames or column names. Start with exposure and claim count. Ask the actuary whether claim amounts or other fields are needed for the stated objective, and suggest additions only when supported by the supplied schemas or metadata.

When multiple sources are used, establish the common observation years or coverage periods and match them at the appropriate policy-period grain. Use only consistently overlapping periods in joined modelling data, and report excluded periods, unmatched records, and reconciliation differences. Do not silently resolve ambiguous period or key alignment.

`01_intake/input/sources.md` is a blank source-register template and `STATE.md`
is a blank run-state template in this structural release. Raw and generated
run data are not included. The existing `02_data_prep/scripts/clean_data.py`
contains reference-schema-specific behavior and must not be assumed to work
for arbitrary inputs.

## Roles
- **Actuary**: the human owner. Reviews data quality, assumptions, challenger signals and the final report, and signs the gates.
- **AI assistant**: proposes assumptions, drafts the report, and may run scripts when the user requests execution and all gates are met. The assumptions register records who proposed and who decided each item.
- **Reviewer**: an independent person or agent for `07_check`.

## Done when
When a run is complete, `06_report/output/` holds its model documentation,
`07_check/output/` holds its independent findings, and `STATE.md` records the
actuary's sign-off and close.
