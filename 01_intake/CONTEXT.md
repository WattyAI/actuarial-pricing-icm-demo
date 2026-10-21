---
type: hybrid
model: local
output: ./output/
---
## Job
Receive and log the data for the pricing workflow. Start from policy/exposure-unit data with an observation period, exposure, and number of claims. The current French motor source is a reference example, not a required dataset.

## Steps
1. Inventory files, formats, sources, dates, periods, units, and known limitations; map available fields to the standard exposure, claim-count, and period concepts.
2. Ask the actuary whether claim amounts or other data are needed for the pricing objective. Review supplied schemas and metadata and suggest relevant available fields or sources for confirmation; do not assume or add them without approval.
3. Record every source and any mapping or period ambiguity in `output/intake_log.md`. Follow applicable privacy and sensitivity requirements; do not assume data are public or non-sensitive.
4. If optional sources are to be combined, identify keys and period fields needed to align them at the policy-period grain. Refer uncertain or ambiguous mappings to the actuary before preparation.

## Done when
The intake record captures what arrived, provenance, mapped concepts, period coverage, caveats, optional-data decisions, and unresolved questions; data-quality findings belong to `02_data_prep`.
