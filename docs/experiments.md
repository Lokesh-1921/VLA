# Experiment ledger

| ID | Base checkpoint | Training data | LoRA | Evaluation split | Status |
| --- | --- | --- | --- | --- | --- |
| A: baseline | Record exact checkpoint | none | none | fixed held-out set | not run |
| B: Bridge-FT | record A checkpoint | BridgeData V2 version | record rank/alpha | same as A | not run |
| C: Bridge-FT + custom | record B checkpoint | Bridge + custom version | record rank/alpha | same as A | not run |

Do not fill metric cells until the linked experiment manifests and prediction files exist. Record hardware, seed, runtime, dataset count, normalizer, commit hash (when available), and failure modes for each run.
