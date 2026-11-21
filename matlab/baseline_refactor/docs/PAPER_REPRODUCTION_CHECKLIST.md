# PeerJ AI Application — completion gate

This is an internal author checklist, not a completed editorial response. It is based on the editorial email and provided manuscript. The journal requested actual code/data, a README and limitations in the conclusion; its separate requests concern image permissions and submission metadata.

| Item | Current status | Required action before release |
| --- | --- | --- |
| Six uploaded MATLAB files | Archived unchanged; corrected CNN branch supplied | Author review, runtime testing and confirm ownership |
| Exact 1,000 original + 1,790 derived images | Missing | Recover files or exact reproducible generation process, verify source identifiers and deposit |
| Historical grouped folds | Missing | Recover assignments and verify that all derivatives stay with parents |
| Internode pretraining/adaptation | Missing | Recover 500 internode image IDs (100 per variety), code, split logic, checkpoint and settings; confirm separation from evaluation |
| ESRGAN | Missing | Recover scripts, checkpoint, scale, preprocessing and derivative generation parameters |
| BA-ViT | Missing | Recover actual MATLAB layer graph, pretrained checkpoint, custom head, training settings and 1,024-D extraction; verify trained components are fold-safe |
| ACO/SSA | Missing | Recover actual code, random seeds, fitness weights, inner validation strategy, cardinality enforcement and masks; use only training folds for selection |
| SVM/KNN | Missing | Recover actual classifiers, standardization, hyperparameters, multiclass strategy, selected feature counts 100/250/500/750/1,000 |
| AlexNet (reported 70.10%) | Missing | Retrieve actual historical implementation; optional new AlexNet code is not a substitute |
| Reported experiment outputs | Partial | Table 4 counts available; recover individual predictions, fold metrics, timing logs and scores where retained |
| README | Baseline documentation supplied | Extend with exact original experiment commands/dependencies and expected outputs |
| Source code/data availability | Not complete | Deposit approved code, data and metadata to suitable repository or supplement; link final version in manuscript |
| Conclusion limitations | Outside code task | Add explicit limitations in manuscript conclusion |
| Figures 1 and 3 | Outside code task | Confirm owner permission and individual file attribution/licence links or remove photos |
| Article type and declarations | Outside code task | Confirm AI Application selection and verify submission-system details |

## Manuscript-specific settings that can be transcribed (not inferred)

Paper pp. 3–5: 384×384 RGB input, 16×16 patches, eight attention heads, 1,024-dimensional feature output; ACO–SSA population 10, maximum 100 iterations, evaporation 0.2, selection threshold 0.5. The paper states grouped fivefold CV and training-only stochastic augmentation. It does not quantify all settings or include all artifacts. These descriptions are not executable source code and must not be silently converted into purported historical implementation.

## Release gate

1. Retrieve original code and artifacts from the experimental authors, or clearly report unavailable items to PeerJ and rerun/revise when necessary. Never fabricate a historical seed, model checkpoint, fold manifest, feature mask or prediction file.
2. Verify data rights, code ownership, dataset variety names, individual figure-file provenance, and exact study subset against the repository record. Approve a licence and remove or mark non-experimental assets.
3. Perform a clean MATLAB R2024b run using the actual pipeline; capture runtime and environment logs, expected outputs, and discrepancies. Distinguish replicated results from newly generated ones.
4. Update the README and manuscript Data Availability, Methods, Results and limitations wherever new evidence or repeat experiments change the reported claims. Deposit immutable versioned source and data. Do not upload this package as the complete BA-ViT/ACO–SSA artifact without completing these steps.
