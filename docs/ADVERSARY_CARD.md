# Task-B structural adversary card

Status: **revised working contract; not yet a release claim.** S05 pose replay
showed that downward camera pitch permits visibly rendered objects whose 3D-box
centers lie outside the formerly inferred ±66.6-degree interval. The interval is
now an explicit benchmark-design gate, not a consequence of visibility or camera
FOV. Candidate-r1 did not enforce it and remains invalidated. A replacement may use
this contract only after independent itemwise pose/AABB replay.

## What is being bounded

The structural adversary observes exactly two action variables from the
user-visible question: the in-place turn direction (`left` or `right`) and its
rotation in degrees. It does not observe the image, scene, target identity or
category, pose, answer-option order, other metadata, or any additional wording.
The action is parsed from the question because the legacy `qb2_orig_turn`
metadata is noncanonical for 180-degree items.

This adversary is narrower than an unrestricted text-only or public-metadata
model. The structural target is therefore a benchmark-construction target, not
a claim that all blind solvers are bounded by it. Full-text, n-gram, learned,
and public-metadata attacks remain mandatory release gates.

## Source-bearing contract and sign convention

- The camera rotates in place; there is no translation.
- The proposed construction envelope requires each target's 3D-box-center
  horizontal bearing in View 1 to lie in
  `(-66.58677555362947, +66.58677555362947)` degrees.
- This exact half-width was encoded by the intended-render targeted miner before
  the replay failure: it is the horizontal center-ray half-angle for a 640×480,
  vertical-FOV-120 pinhole camera. We retain that prior design choice as an
  explicit target-center eligibility cone, not as a consequence of visibility.
- The nearest nominal support transition is at 75 degrees, leaving 8.4132 degrees
  of design margin and 3.4132 degrees after the prespecified ±5-degree action
  calibration perturbation. A wider cone would be an outcome-informed redesign.
- This is a task-eligibility cone, not camera HFOV. Visibility, masks, or a pitched
  image alone do not establish membership.
- Every replacement item must pass both independent bearing replay and separate
  mask-based source-visibility/identifiability gates.
- Positive bearing means to the observer's right.
- A left camera turn by `r` shifts the stationary target-bearing interval by
  `+r`; a right turn shifts it by `-r`.
- Four 90-degree answer zones are centered at front 0, right +90, behind 180,
  and left -90 degrees.
- A boundary-only intersection has zero angular measure and does not enter the
  population support. Candidate construction already applies a 15-degree
  source-boundary margin, so released items must not depend on this convention.

The recovered legacy label helpers disagree on ownership of exactly +45 and
-45 degrees. This does not alter positive-measure supports and cannot affect the
current candidate because its recorded object extents are at least 16.58 degrees
from a zone boundary. S05 must nevertheless replace those helpers with one
canonical half-open assignment and verify every gold label from pose geometry.

The resulting supports are:

| Rotation | Left-turn support | Right-turn support | Size |
|---:|---|---|---:|
| 30 | front, right | front, left | 2 |
| 60 | front, right | front, left | 2 |
| 90 | front, right, behind | front, left, behind | 3 |
| 120 | right, behind | left, behind | 2 |
| 180 | right, behind, left | right, behind, left | 3 |

The 180-degree supports are identical for left and right, although direction
remains part of the declared action cell.

## Population and finite targets

For action cell `z`, let `S(z)` be the support derived before inspecting gold
labels, `k_z = |S(z)|`, and `p(z)` its population weight. At least one label in
the cell has conditional probability at least `1/k_z`; therefore the minimum
attainable accuracy of the best constant-per-action-cell adversary is

`sum_z p(z) / k_z`.

For a finite cell containing `n_z` examples, pigeonhole gives the integer lower
bound `ceil(n_z/k_z)`. It is attainable by assigning support-label counts that
differ by at most one. The finite construction target is therefore

`sum_z ceil(n_z/k_z)` correct predictions.

If a replacement contains 114 examples in each of ten direction/rotation cells,
Six cells have support size two and four have support size three, yielding

- population target: `(1/2 + 1/2 + 1/3 + 1/2 + 1/3) / 5 = 13/30`;
- finite target: `6*ceil(114/2) + 4*ceil(114/3) = 494`;
- finite accuracy: `494/1,140 = 13/30 = 43.333...%`.

Candidate-r1 realizes those empirical counts but violates the source-envelope
precondition for 304 reservoir-sourced rows. Its 494/1,140 score is therefore an
audit statistic, not evidence that the conditional geometric target applies.

These statements require all released gold labels to be in their analytically
reachable support and each realized action cell's largest label count to equal
its finite target. The audit fails on an invalid label, deterministic support,
duplicate item, action/metadata rotation mismatch, or positive excess.

## Independent verification and sensitivity

`egoshift.support` uses analytic interval intersections. A separate executable,
`tools/independent_support_enumerator.py`, imports none of that code and instead
enumerates 200,000 interior source bearings per action. Both implementations
must agree on every declared configuration.

Sensitivity configurations vary the declared source-bearing envelope, including
the old intended-render center-ray half-width of 66.5868 degrees, shift all
zone boundaries, and replace the action set. A ±45-degree envelope is a
prespecified negative control: it creates deterministic cells and demonstrates
that the formula is not automatically favorable. Sensitivity results characterize
the design; they do not retrofit supports after inspecting benchmark labels.

## Claims this card does not license

- that 13/30 is the best unrestricted no-image accuracy;
- that arbitrary textual, template, category, scene, or metadata leakage is absent;
- that every action-conditioned benchmark has the same support or target;
- that the elementary pigeonhole result is itself a deep mathematical theorem;
- that model performance below 13/30 proves an absence of spatial ability.
