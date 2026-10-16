# Historical-exposure ledger contract

Candidate construction fails closed unless physical split roles bind an
independently reviewed historical-exposure audit. This prevents a technically
valid split from being mislabeled “fresh” merely because an evaluated pool was
omitted from the input list.

The audit must satisfy `schemas/exposure_ledger.schema.json` and contain:

- `audit_id = egoshift_historical_exposure_ledger`;
- `status = complete_reviewed`, `complete = true`, and `pass = true`;
- every historical model, blind/control, human, pilot, survey, and manually
  inspected QA manifest relevant to Tasks A/B/C;
- an exact SHA-256 for every file;
- `reviewed_manifest_count` equal to the number of unique file paths; and
- `manifest_sha256` equal to the canonical digest computed by
  `egoshift.split_roles.canonical_exposure_manifest_sha256`.

`tools/build_split_roles.py` independently rehashes every listed file and binds
the reviewed manifest digest plus the audit-file hash into its output. All
Task-B/Task-C capacity, mining, merge, and selection consumers call
`require_complete_split_roles`; `pass=true` without this binding is rejected.

Completeness is a scientific review decision, not something a directory glob
can certify. Generate the inventory broadly, then independently check it against
run scripts, result directories, human-study manifests, documentation, and
manually viewed pools. Treat unknown or uncertain exposure conservatively as
exposed.

The current `work/candidate_r2/audits/split_roles.json` predates this contract
and is intentionally rejected. Do not patch it in place; build a new version
from the reviewed ledger and regenerate every downstream artifact.
