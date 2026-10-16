# FAQ

### What is an "action-only shortcut"?

A question that states a camera turn gives information before the image is
considered. A 180° turn cannot put an object where a 30° turn would. So a
predictor that sees only the stated turn, and nothing else, already beats
chance. If a benchmark does not measure that, part of its reported accuracy is
not visual reasoning.

### Is this just answer balancing?

No, and the distinction is the paper's main point. Balancing equalises answers
**across the whole dataset**. The shortcut operates **within each action cell**.
A set can be perfectly balanced globally and still let a turn-only predictor
score far above chance. An earlier build of our own data was balanced by the
usual standard and leaked **72.3%**.

### Why is the ceiling computed rather than measured?

A measured shortcut is whatever the attacks you happened to run could find. A
computed ceiling is what the geometry permits, known before any model runs.
Both are useful and they answer different questions; we use ours alongside
learned attacks, not instead of them. The learned-attack battery has a protocol
fixed in advance, stated in the supplementary material.

### Does attaining the ceiling mean the questions have no shortcuts?

**No.** The ceiling bounds exactly one shortcut: answering from the stated turn
with no image. It says nothing about wording, metadata, or a cue that some
future model finds and we did not anticipate. The paper does not claim the
questions are free of shortcuts in general, and neither should anyone using it.

### Why do models fail at 90° and 180° but not 120°?

A 120° turn is answered correctly more than twice as often as a 90° one. The
responses are consistent with a rotation applied to an *assumed* straight-ahead
bearing rather than to the bearing in the image: at 90° and 180° that assumption
and the truth diverge most. This is an interpretation of the evidence, not an
established mechanism — the matched counterfactuals that would settle it have
not been run, and a model behaving this way for another reason would produce
the same evidence.

### Can I report a single pooled number?

Please do not. It is the specific failure this benchmark documents. Pooled Task-B
accuracy puts the best configuration at 83.6% against a trained-human 84.6%,
which reads as nearly solved; per turn, the same system trails the panel badly
at 90° and 180°. Report per stated turn, against the per-cell ceiling.

### Why is Task A called a control?

It checks that a model can find a boxed object and match it across two views at
all. It carries no difficulty claim, and two selection questions about it remain
unsettled, recorded in the supplementary material. No Task-A number enters a
composite score.

### 180° rests on how many questions?

56 of the 1,098 held-out Task-B questions, because that geometry is scarce. The
paper reports the two hard turns together as well — 38.0% over 290 questions,
95% scene-clustered interval [32.9, 43.1] — so the conclusion does not rest on
the smallest cell.

### Do I need AI2-THOR?

Only to regenerate frames. Scoring, auditing and verification run on question
and prediction files alone.

### How do I get held-out answers?

You do not; they are withheld so the split stays a test set. See
[`data.md`](data.md) for submitting predictions.
