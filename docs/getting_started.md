# Getting started

Everything below runs against the bundled six-item fixture in `sample/`, so it
works immediately after installation with no download. The fixture is schema
and scoring smoke data: it contains no simulator image and is **not** evidence
for any result in the paper.

## Score predictions

```shell
egoshift score \
  --questions   sample/questions.jsonl \
  --predictions sample/predictions.jsonl
```

Scoring is deterministic and reports missing, extra and unparseable predictions
explicitly rather than silently dropping them. Add `--strict` to exit non-zero
when any of those counts is above zero.

The prediction format is one JSON object per line with a question id and a
chosen option. `schemas/prediction.schema.json` is the contract.

## Validate a dataset manifest

```shell
egoshift validate-manifest sample/manifest.json --root sample
```

Checks the manifest against `schemas/dataset_manifest.schema.json`, confirms
every path is relative and inside the root, and verifies the SHA-256 of each
listed file. Content addressing is what lets a result record name exactly the
bytes it was computed from.

## Audit Task B's action cells

This is the central check of the paper, and the one to run on any new or
modified split:

```shell
egoshift audit-taskb --questions path/to/taskb_questions.jsonl
```

For each of the ten cells `z = (turn size, direction)` it derives `S(z)`, the
set of answers the geometry permits, counts the questions in the cell, and
checks that the commonest answer occupies exactly `ceil(n_z / |S(z)|)` of them.
If any cell sits above its bound, the action-only ceiling is higher than
claimed and the audit fails.

Useful options:

| Option | Meaning |
|:---|:---|
| `--expected-sha256` | refuse to audit a file whose content hash is not this |
| `--expected-rotations` | the turn sizes that must be present (default `30,60,90,120,180`) |
| `--expected-items-per-cell` | assert an exact per-cell count |
| `--source-bearing-half-width` | the View-1 bearing gate in degrees. This is **not** the camera HFOV; it is the construction constraint on where a target may start |

## Run the whole check suite

```shell
make check          # pytest suite + smoke + support cross-check
make test           # pytest only
make smoke          # fixture end-to-end
make support-check  # the two S(z) implementations must agree
```

`make support-check` is the one worth understanding. `S(z)` is computed twice,
by an interval derivation and by a dense non-sharing enumeration written
independently, and the build is accepted only when they agree. A ceiling is a
claim about what a shortcut can reach, so a single implementation of it is a
single point of failure.

## Evaluating your own model

1. Produce one prediction per held-out question, in the schema above.
2. Score the development split locally to check your parser.
3. **Report accuracy per stated turn, not pooled.** The failure this benchmark
   exists to expose is invisible in a pooled number.
4. Report against the action-only ceiling for each cell. A Task-B score below
   45.90% on the held-out split is at or under what the stated turn alone
   reaches.

Held-out answers are withheld. See [`data.md`](data.md).
