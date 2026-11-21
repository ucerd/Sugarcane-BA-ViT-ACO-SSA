# Changes from supplied MATLAB files

- Fixed missing dataset-root argument and replaced hard-coded placeholder paths with function parameters.
- Replaced unverified train/test splitting with explicit grouped CV and an opt-in for new seeded folds.
- Stopped passing the held-out test set as training validation data.
- Fixed swapped precision/recall denominators; added specificity, F1, G-mean and pooled summaries.
- Aligned prediction scores to explicit class labels; compute ROC from actual FPR rather than index positions.
- Added output logs, configuration capture, fixed seeds, score exports and optional network checkpoints.
- Added optional new AlexNet comparator, strictly separate from the reported historical experiment.
- Added tests, dataset manifest schema, manuscript arithmetic audit, provenance archive and release checklist.
