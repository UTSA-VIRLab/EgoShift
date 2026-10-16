# Installation

Two levels, depending on what you need. **Scoring and auditing a released split
needs only the core package.** The simulator is required solely to regenerate
frames from scratch.

## 1. Core package — scoring, auditing, verification

Python 3.10 or newer.

```shell
git clone https://github.com/UTSA-VIRLab/EgoShift.git
cd EgoShift
python -m pip install -e .
```

Verify the install:

```shell
make check
```

That runs the unit suite, the end-to-end smoke test on the bundled fixture, and
the support cross-check that compares the two independent implementations of
`S(z)`. A clean run ends with `150 passed, 4 skipped` and exit status 0. The
four skips are tests that assert properties of generated tracker documents,
which are produced in the private working tree and are not shipped here.

## 2. Pinned environment

For a reproduction that matches the paper's software versions:

```shell
conda env create -f environment.yml
conda activate egoshift
python -m pip install -e . -c constraints.txt
```

`constraints.txt` pins the transitive set. `environment.yml` pins the
interpreter and the compiled dependencies.

## 3. Regenerating frames (optional)

Only needed to rebuild the rendered views. Requires a GPU and a display server
or Xvfb.

```shell
pip install ai2thor prior
```

- [AI2-THOR](https://ai2thor.allenai.org/) — the simulator and its built-in
  asset library, Apache-2.0.
- [`prior`](https://github.com/allenai/prior) — delivers the ProcTHOR-10k
  houses, Apache-2.0.

The paper's frames were rendered at 640x480. Rendering is the only step that
needs the simulator; every audit, score and verification in this repository
runs on the question and prediction files alone.

## Troubleshooting

**`make check` fails with `No module named 'tools'`.** Run it from the
repository root. The Makefile sets `PYTHONPATH=src:.` and the tests import
`tools.*` relative to that root.

**`unittest discover` reports fewer tests than `make test`.** Expected. The
suite mixes `unittest.TestCase` classes with pytest-style module-level tests,
and `unittest discover` collects zero tests from the latter without reporting
it. `make test` uses pytest and is the authoritative run.

**AI2-THOR cannot start.** It needs an X display. Use `xvfb-run -a` or
AI2-THOR's CloudRendering platform. This affects frame regeneration only.
