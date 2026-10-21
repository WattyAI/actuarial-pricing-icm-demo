---
type: hybrid
model: local
inputs: [ ../06_report/output/, ../07_check/output/, ../08_client_review/output/ ]
output: ./output/
---
## Job
Assemble and record the closing package for this run: a package, not loose
files.

## Steps
1. **Do not copy deliverables into `99_close/output/`.** The producing stage's `output/` stays the only copy in this workspace. Duplicated copies with no designated original are how a delivered figure and its record come to disagree.
2. Author the package's own files directly in `output/`: a concise `SUMMARY.md` and an index listing this run's deliverables with their paths and SHA-256 hashes. The deliverable list is run-specific.
3. Update `STATE.md` to done.

## Done when
`output/SUMMARY.md` exists, the index lists every applicable deliverable with its hash, and `STATE.md` reflects close.
