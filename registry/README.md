# Result registry

Final result records will be immutable JSON files validated against
`schemas/result_record.schema.json`. Each record must identify the dataset manifest,
raw predictions, model revision, prompt/parser protocol, scorer, statistical code,
metrics, and paper table/figure/claim consumers by SHA-256.

Candidate and smoke outputs do not enter the final registry. No raw human identifiers,
provider credentials, private paths, or unapproved proprietary responses are allowed.
