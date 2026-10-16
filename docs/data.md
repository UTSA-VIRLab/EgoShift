# Data

## What is released

| Component | Terms | Status |
|:---|:---|:---|
| Questions, options, labels, scene metadata, audit outputs | **CC-BY-4.0** | released |
| Rendered frames | upstream **Apache-2.0** (AI2-THOR, ProcTHOR-10k), attribution and `NOTICE` preserved | released |
| Development-split answers | CC-BY-4.0 | released |
| **Held-out answers** | — | **withheld** |

Held-out answers stay withheld so the split remains usable as a test set. The
held-out *questions* are public; only their answers are not.

## Composition

Held-out, which is what results are reported on:

| | A: same object? | B: where after a turn? | C: still visible? |
|:---|:---:|:---:|:---:|
| Role | control | **primary** | supporting |
| Held-out questions | 5,838 | 1,098 | 5,850 |
| Development (public) | 3,582 | 440 | 3,705 |
| Rooms | 3,119 | 736 | 1,334 |
| Object categories | 28 | 45 | 49 |
| Largest category's share | 28% | 5% | 13% |

Development and held-out questions are drawn from **separate rooms**, so a
model cannot have seen a held-out room during development.

## Provenance

Every scene comes from [ProcTHOR-10k](https://procthor.allenai.org/) rendered
with [AI2-THOR](https://ai2thor.allenai.org/). A provenance sweep over all
9,838 staged houses found 1,465 distinct asset ids, every one an AI2-THOR
built-in library asset, and zero external or Objaverse-style identifiers. Both
upstream projects are Apache-2.0.

Because the scenes are simulator-rendered, object categories follow the
AI2-THOR asset library. Neither the objects nor the room layouts represent real
interiors, and nothing here shows that results transfer to natural images. This
is stated as a limitation in the paper and it is a real one.

## Human responses are not released

The trained-human reference is reported as aggregates only. Per-rater
responses, the pseudonym map and all intake files stay in a private working
tree and are excluded from this repository by `.gitignore` and by
`tools/build_reviewer_bundle.py`, which names them explicitly as exclusions.

The institution's review board determined before collection that the annotation
study does not constitute human-subjects research requiring oversight. That is
a **non-human-subjects-research determination**, not an approval and not an
exemption, and it should not be described as either.

## Getting the data

The question files and development answers are distributed with the release
bundle. Rendered frames are large and are hosted separately rather than in git.

> **Note.** The data host is being set up for the camera-ready. Until the link
> appears here, use `sample/` for the schema and the scoring path, and open an
> issue if you need early access.

## Submitting for held-out scoring

Produce one prediction per held-out question against
`schemas/prediction.schema.json`, verify it scores cleanly on the development
split, and open an issue. Report per stated turn, never pooled.
