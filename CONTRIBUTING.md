# Contributing

Three kinds of contribution are useful here, in descending order of how much
we want them.

## 1. Leaderboard submissions

Run a system and send the predictions. Open a
[results submission](https://github.com/UTSA-VIRLab/EgoShift/issues/new?template=leaderboard.yml).

Held-out answers are withheld, so we score the file rather than trusting a
reported number. Check it parses against the development split first:

```shell
egoshift score --questions dev_questions.jsonl --predictions yours.jsonl --strict
```

**Per-turn numbers are required.** A pooled score cannot be interpreted — that
is the finding the benchmark exists to report, and a pooled row on the
leaderboard would reproduce the error we are documenting.

## 2. Attacks on the construction

The strongest contribution is showing the floor is wrong.

The action-only ceiling bounds exactly one shortcut: answering from the stated
turn with no image. It says nothing about wording, metadata, or a cue we did
not anticipate. If you find a predictor that beats the ceiling using only
public fields, that is a defect in the benchmark and we want the issue.

Useful starting points: `egoshift audit-taskb` recomputes the ceiling per cell,
and `tools/compare_support_implementations.py` checks the two independent
derivations of `S(z)` against each other.

## 3. Code and documentation

Normal pull requests. Before opening one:

```shell
pip install -e ".[test,full]"
make check
```

`make check` runs the test suite, the end-to-end smoke test on the bundled
fixture, and the support cross-check. It must pass. If you change anything
that affects a label, say so explicitly in the PR — a silent label change
invalidates every published number.

### Conventions

- Tests go in `tests/`. The suite mixes `unittest.TestCase` classes with
  pytest-style module-level tests; **run it with pytest**, because
  `unittest discover` collects zero tests from the latter without reporting it.
- Geometry changes need a test that fails before the change and passes after.
  Several defects in this codebase were sign errors that a weak assertion let
  through, so an assertion that cannot fail is worse than none.
- Keep `schemas/` and the code in step. The schemas are the contract.

## Reporting a bad question

Labels are computed from simulator geometry, not annotated by hand, so a
dispute is usually about whether an item is *answerable* rather than about the
key. Both are worth reporting: open a
[question dispute](https://github.com/UTSA-VIRLab/EgoShift/issues/new?template=data.yml)
with the question id.

We do not change a question in response to a score. We do change one that is
geometrically ill-posed, and we say so in the changelog.
