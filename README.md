# EgoShift: Bounding the Action-Only Shortcut in Egocentric Spatial Updating for Vision–Language Models

<p align="center">
  <a href="https://utsa-virlab.github.io/EgoShift/"><strong>Project Page</strong></a> &nbsp;|&nbsp;
  <a href="assets/EgoShift_WACV2027.pdf"><strong>Paper</strong></a> &nbsp;|&nbsp;
  <a href="assets/EgoShift_WACV2027_supplementary.pdf"><strong>Supplementary</strong></a> &nbsp;|&nbsp;
  <a href="#installation"><strong>Installation</strong></a> &nbsp;|&nbsp;
  <a href="#getting-started"><strong>Getting Started</strong></a>
</p>

**News: Our paper is accepted at WACV 2027 (Evaluation & Datasets track).**

[Faisal Ahmed](https://www.linkedin.com/in/faisal-ahmed-565943145/), [John Quarles](https://www.linkedin.com/in/johnpquarles/), [Kevin Desai](https://www.linkedin.com/in/kevinpdesai/) — Department of Computer Science, University of Texas at San Antonio

A model can score well on egocentric spatial reasoning **without ever looking at the image**.
When a question states a camera turn, that turn alone rules out some answers, so a predictor given
only the stated action already beats chance by a wide margin. Balanced answers, text-only arms and
human references do not catch this, because they are not conditioned on the action.

**EgoShift makes the shortcut a quantity instead of a worry.** The camera geometry determines which
answers each turn can produce, so the highest score reachable from the action alone can be *computed
before any model is run*. We derive that ceiling and build every turn to sit exactly on it.

<p align="center"><img src="./resources/tasks.png" alt="the three EgoShift tasks" width="94%"></p>

The audit is not hypothetical. An earlier build of our own data was balanced by the usual standard,
and a predictor that saw **only the stated turn** scored **72.3%** on it.

### Why it matters

Across 36 systems, pooled accuracy looks close to solved while a severe failure hides inside it.
Accuracy does **not** fall as the turn grows larger — it collapses at exactly two turns:

| Stated turn | 30° | 60° | 90° | 120° | 180° |
|:---|:---:|:---:|:---:|:---:|:---:|
| Frontier mean (Task B) | 86.8 | 82.1 | **38.3** | 84.3 | **36.8** |

At 90° and 180° no frontier configuration exceeds 50%, and most return the same answer they give
with no image at all. A single pooled number reports progress that has not happened.

<p align="center"><img src="./resources/rotation_collapse.png" alt="per-turn collapse" width="82%"></p>

### The benchmark

12,786 held-out questions in simulated indoor scenes, over three tasks. Development and held-out
questions come from **separate rooms**, and all three tasks carry a trained-human reference.

| | A: same object? | B: where after a turn? | C: still visible? |
|:---|:---:|:---:|:---:|
| Role | control | **primary** | supporting |
| Held-out questions | 5,838 | 1,098 | 5,850 |
| Development (public) | 3,582 | 440 | 3,705 |
| Rooms, held out | 3,119 | 736 | 1,334 |
| Object categories | 28 | 45 | 49 |
| Action-only ceiling | — | **45.90%** | — |

Task B's held-out ceiling is `504/1,098 = 45.90%`: the fewest questions the commonest answer can
cover in each of the ten action cells, summed. The build matches it in **every** cell.

### Installation

Python 3.10 or newer. The core package has no simulator dependency — it scores, audits and verifies.

```shell
git clone https://github.com/UTSA-VIRLab/EgoShift.git
cd EgoShift
python -m pip install -e .          # core: no required dependencies
python -m pip install -e ".[test,full]"   # + pytest, numpy, scipy, Pillow
make check
```

Or with conda, pinned:

```shell
conda env create -f environment.yml && conda activate egoshift
python -m pip install -e . -c constraints.txt
```

Regenerating frames additionally needs [AI2-THOR](https://ai2thor.allenai.org/) and the
[`prior`](https://github.com/allenai/prior) package for ProcTHOR-10k houses. Scoring and auditing a
released split do **not** require them. See [`docs/installation.md`](docs/installation.md).

### Getting Started

Score a prediction file against a question file:

```shell
egoshift score \
  --questions   sample/questions.jsonl \
  --predictions sample/predictions.jsonl
```

Run the end-to-end smoke test on the bundled six-item fixture, without installing:

```shell
PYTHONPATH=src python -m egoshift smoke --root sample
PYTHONPATH=src python -m unittest discover -s tests -v
```

Audit Task B's action cells against the analytic geometry support — this is the
ceiling check, and it fails loudly if any cell does not sit on the bound:

```shell
egoshift audit-taskb --questions path/to/taskb_questions.jsonl
```

Validate a dataset manifest: schema, safe relative paths and SHA-256 content hashes.

```shell
egoshift validate-manifest sample/manifest.json --root sample
```

The fixture is schema and scoring smoke data only. It contains no simulator image and is not
evidence for the paper. Full walkthrough: [`docs/getting_started.md`](docs/getting_started.md).

### How the ceiling is computed

Write `z = (r, d)` for the stated turn, `r` its size and `d` its direction. Five sizes and two
directions give ten **cells**. A predictor given `z` and nothing else — no image, no scene, no
object, no wording — can do no better than answering the commonest answer in each cell. With `S(z)`
the answers the geometry allows and `n_z` the questions in cell `z`:

```
C_min = (1/N) · Σ_z  ⌈ n_z / |S(z)| ⌉
```

Two independent implementations compute `S(z)` — an interval derivation and a dense enumeration —
and the build is accepted only when they agree and every cell attains the bound. Balancing answers
*across the whole set* does not achieve this, which is why the earlier 72.3% build looked fine.

| In the paper | In the code |
|:---|:---|
| action-only ceiling and the per-cell check (Sec. 3) | [`taskb_audit.py`](src/egoshift/taskb_audit.py), CLI `egoshift audit-taskb` |
| the answer set `S(z)`, interval derivation | [`support.py`](src/egoshift/support.py) |
| dense non-sharing enumeration verifier | [`taskb_geometry.py`](src/egoshift/taskb_geometry.py) |
| deterministic multiple-choice scoring | [`scoring.py`](src/egoshift/scoring.py) |
| fail-closed split roles, exposure ledger | [`split_roles.py`](src/egoshift/split_roles.py) |
| manifest schema and content addressing | [`manifest.py`](src/egoshift/manifest.py), [`hashing.py`](src/egoshift/hashing.py) |
| integrated construction audit | [`integrated_audit.py`](src/egoshift/integrated_audit.py) |

### Results

Pooled Task B accuracy puts the best configuration at **83.6%** against a trained-human mean of
**84.6%** — which reads as nearly solved. Read one turn at a time, those two numbers stop matching:
the best model beats the panel at 30°, 60° and 120°, and trails it badly at 90° and 180°.

Because the 180° cell holds only 56 questions, we also report the two hard turns together:
**38.0%** over 290 questions, 95% scene-clustered interval **[32.9, 43.1]**.

<p align="center"><img src="./resources/qualitative.png" alt="qualitative example" width="88%"></p>

**Progress on EgoShift should appear as higher scores at 90° and 180°.** A system that raises its
pooled score while leaving those at the action-only ceiling has not improved at the thing the
benchmark measures.

### Data

Questions, labels and audit outputs are released under **CC-BY-4.0**. Rendered frames are
redistributed under the upstream **Apache-2.0** terms of AI2-THOR and ProcTHOR-10k, with attribution
and `NOTICE` preserved. **Held-out answers are withheld** so the split stays usable as a test set;
submit predictions to score against them. See [`docs/data.md`](docs/data.md); the full datasheet
is Section F of the [supplementary material](assets/EgoShift_WACV2027_supplementary.pdf).

### Repository layout

| Path | What is in it |
|:---|:---|
| [`src/egoshift/`](src/egoshift) | scoring, geometry, ceiling derivation, split roles |
| [`schemas/`](schemas) | dataset-manifest, prediction and result-record schemas |
| [`tools/`](tools) | model panel runners, bundle builders, audit and anonymity checks |
| [`sample/`](sample) | six-item fixture, dummy predictions, expected metrics |
| [`tests/`](tests) | unit tests; `make check` runs them |
| [`configs/`](configs) | task and panel configurations |
| [`docs/`](docs) | [installation](docs/installation.md), [getting started](docs/getting_started.md), [data](docs/data.md), [FAQ](docs/faq.md) |
| [`assets/`](assets) | paper and supplementary PDFs |

### Citation

```bibtex
@inproceedings{ahmed2027egoshift,
  title     = {EgoShift: Bounding the Action-Only Shortcut in Egocentric
               Spatial Updating for Vision--Language Models},
  author    = {Ahmed, Faisal and Quarles, John and Desai, Kevin},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

### License

Code under the licence in [`LICENSE`](LICENSE). Data terms and third-party attribution are in
[`NOTICE`](NOTICE) and [`LICENSE_STATUS.md`](LICENSE_STATUS.md). EgoShift is built on
[AI2-THOR](https://ai2thor.allenai.org/) and [ProcTHOR-10k](https://procthor.allenai.org/),
both Apache-2.0.

### Acknowledgments

Supported by the National Science Foundation. We thank the paid annotators of the San Antonio
Virtual Environments (SAVE) lab for the trained-human reference.
