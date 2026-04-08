# Project Instructions

## Test Data Group Names
- Never use real release-group names in test fixtures, examples, snapshots, or docs.
- Use placeholder group names only, such as `CancelHD`, `CancelWEB`, `CancelGroup`.
- Do not include any concrete real group-name examples in this repository.

## Commit-Time Check
- Before each commit, review staged diffs (`git diff --cached`) and ensure no real group names appear.
- If any real group name is found, replace it with a placeholder before committing.
