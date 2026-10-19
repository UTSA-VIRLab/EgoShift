# Changelog

Benchmark content is versioned separately from the code. Any change that could
move a published number is called out explicitly.

## v1.0 — 2026-10-17

First public release, accompanying the WACV 2027 paper.

**Benchmark.** 12,786 held-out questions over three tasks. Task-B action-only
ceiling `504/1,098 = 45.90%`, attained in all ten action cells. Development and
held-out questions drawn from disjoint rooms.

**Released.** Construction, scoring and audit code; schemas; a six-item
fixture; the full model panel's raw completions; build and audit reports;
human aggregates. Dataset on
[Hugging Face](https://huggingface.co/datasets/utsavirlab/EgoShift) and
[Zenodo](https://doi.org/10.5281/zenodo.23440746).

**Withheld.** Held-out answers, all bearing geometry for held-out items, and
every per-rater human response.

**Known open.** The residual-shortcut battery, the remaining presentation
controls and the action-necessity counterfactuals have not run; their protocols
were fixed in advance and are in the supplementary material. Task A is a
positive control with two unsettled selection questions and carries no
difficulty claim.

*No label has changed since the evaluation reported in the paper.*
