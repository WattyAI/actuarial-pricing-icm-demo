---
type: hybrid
model: frontier
inputs: [ ../02_data_prep/output/ ]
output: ./output/
---
## Job
Set out and record the assumptions this run will use. Derive proposals from
the pricing objective, mapped data, and data-quality findings; flag every
judgment call explicitly rather than silently defaulting.

## Steps
1. Review the prepared population, available fields, periods, and limitations.
   Ask the actuary whether additional data or candidate fields suggested by
   the supplied schema should be considered; do not assume they are required
   or approved.
2. Propose only assumptions supported by the data, model purpose, or an
   explicitly documented external basis. Named factors, distributions, bands,
   caps, offsets, and loadings from the French motor reference run are not
   defaults for other data.
3. Record each proposal, rationale, evidence, sensitivity treatment, proposer,
   decision-maker, and caveat in `output/Assumptions_and_Judgements.md`.

## Done when
The assumptions register for this run is ready for modelling and the
`02_data_prep` gate has passed (see that stage's `CONTEXT.md`).
