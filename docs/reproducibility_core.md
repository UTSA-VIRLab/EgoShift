# EgoShift reproducibility core

This directory is the clean-room reviewer-artifact skeleton. It is intentionally
separate from the legacy working source, 292-GB data/cache tree, model outputs, and
human-response files.

Current scope:

- safe relative paths and SHA-256 content addressing;
- dataset-manifest, prediction, and result-record schemas;
- deterministic multiple-choice scoring with explicit missing/extra/parse counts;
- result lineage fields for dataset, prediction, prompt, model, scorer, statistics,
  and paper consumers;
- analytic action-support derivation, a non-sharing dense enumeration verifier,
  and finite action-cell construction audits;
- fail-closed physical split roles bound to a separately reviewed, hash-pinned
  historical-exposure ledger;
- Task-B/Task-C S05 candidate-construction and live-evidence infrastructure;
- a synthetic six-item fixture, dummy predictions, expected metrics, unit tests, and
  CI/local commands.

The fixture is schema/scoring smoke data only. It contains no simulator image and is
not evidence for the paper. Candidate benchmark files remain outside this package
until S07 freezes them, and redistribution remains unauthorized until external license
gate U6 is resolved.

## Quick start

From this directory:

```bash
python -m pip install -e .
make check
```

Without installing:

```bash
PYTHONPATH=src python -m egoshift smoke --root sample
PYTHONPATH=src python -m unittest discover -s tests -v
```

Score a prediction file:

```bash
egoshift score \
  --questions sample/questions.jsonl \
  --predictions sample/predictions.jsonl
```

Validate a content-addressed manifest:

```bash
egoshift validate-manifest sample/manifest.json --root sample
```

Audit a Task-B candidate without modifying it:

```bash
egoshift audit-taskb \
  --questions /path/to/task_b.jsonl \
  --expected-sha256 EXPECTED_SHA256 \
  --expected-items-per-cell EXPECTED_ITEMS_PER_ACTION_CELL
python tools/independent_support_enumerator.py \
  --config configs/taskb_geometry_sensitivity.json
```

The structural adversary, explicit source-bearing-envelope assumptions, proof
sketch, conditional 13/30 construction target, candidate-r1 invalidation, and
claim limits are documented in `docs/ADVERSARY_CARD.md`.

The reviewed exposure binding required before any real S05 build is documented
in `docs/EXPOSURE_LEDGER_CONTRACT.md`. As of 2026-08-20, no real candidate-r2
manifest exists; construction deliberately stops on the incomplete historical-
exposure ledger.

No command contacts a network service. The runtime package uses only the Python
standard library.
