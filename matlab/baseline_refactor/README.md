# Bud-Based Sugarcane Variety Identification — MATLAB supplementary code

**Manuscript:** Bud-Based Sugarcane Variety Identification Using a Bud Attention Vision Transformer and Hybrid ACO–SSA Feature Selection  
**Journal:** PeerJ Computer Science (proposed AI Application category)  
**Manuscript ID:** 152775  
**Manuscript authors:** Tassadaq Hussain, Soltan Alharbi, Mukhtiar Bano, Kim Geok Tan

**Release status: author review required; incomplete manuscript reproduction.** This repository refactors the six supplied MATLAB files for ResNet-50 and MobileNetV2. It adds grouped cross-validation, an optional newly implemented AlexNet comparator, reproducibility logging, and independent arithmetic checks of the manuscript's confusion matrix. Neither the original BA-ViT/ACO–SSA implementation nor the experimental images, checkpoints, historical folds or original AlexNet code was supplied. The revised CNN code is not the implementation that generated the reported 94.10% accuracy. Do not describe it as such. No new training results are supplied. See `docs/PAPER_REPRODUCTION_CHECKLIST.md` before submission.

## 1. Project description

The manuscript studies recognition of five sugarcane varieties from bud photographs. Its reported workflow combines ESRGAN, BA-ViT feature extraction, ACO–SSA feature selection and SVM/KNN classifiers. The runnable code here is a **separate supplemental CNN baseline workflow**, not a reconstruction of that historical pipeline. The original uploads are retained unchanged in `original_upload/` for provenance and are not on the runtime path. The new code addresses leakage at the source-photo level, not at the plant level.

## 2. Dataset information

The manuscript cites **Sugarcane-PK v1.0: A Scale-Referenced Dataset and MATLAB Implementation for Precision Phenotyping of Pakistani Sugarcane Varieties**, Hussain (2026), Zenodo, https://doi.org/10.5281/zenodo.18619387. It reports a CC BY 4.0 licence (https://creativecommons.org/licenses/by/4.0/). The repository record, file paths, rights ownership and exact subset must be checked by the authors; they were not independently verified in this package.

The paper describes a parent archive of 7,362 images across eight morphological categories and an experimental working set of 2,790 images consisting of 1,000 originals and 1,790 derived views. The five classes and reported distribution are:

| Variety | Originals | Working set |
| --- | ---: | ---: |
| CP77-400 | 200 | 563 |
| CPF-237 | 200 | 560 |
| CPF-250 | 200 | 550 |
| CPF-251 | 200 | 553 |
| CPF-253 | 200 | 564 |
| Total | 1,000 | 2,790 |

No biological image files are bundled. Download the parent archive only from its verified record. Ask the experiment owner for the exact study subset and derivative images (or exact reconstruction code and settings). Do not choose an arbitrary 1,000-image subset and label it as the paper's data. The old `split` field alone cannot establish grouped historical folds.

Place images in a local dataset directory and prepare a CSV manifest according to `data/README.md`. The example CSV contains headers only; it is not usable experimental data. Paths are relative to the dataset directory. Each original and its derivatives must share a verified `source_image_id`. An original row has `image_id == source_image_id`. Provide the actual historical fold column for the paper dataset; generating folds is allowed only for explicitly new baseline experiments. Use a unique identifier for each image and retain original archive paths/metadata separately.

## 3. Software requirements

Target: MATLAB R2024b (the release stated in the manuscript). The refactor retains the `trainNetwork` API for compatibility with the original scripts. Requirements:

- MATLAB, Deep Learning Toolbox, Statistics and Machine Learning Toolbox, Image Processing Toolbox.
- Deep Learning Toolbox Model for ResNet-50 Network and Model for MobileNet-v2 Network add-ons (pretrained weights).
- Optional Deep Learning Toolbox Model for AlexNet Network add-on for the new AlexNet baseline.
- Optional Parallel Computing Toolbox and supported NVIDIA GPU with an appropriate driver for GPU training. CPU execution is supported conceptually but may be slow.
- Sufficient storage for data and per-fold networks; use `cfg.saveNetworks = false` to omit networks, while keeping predictions and logs.

The manuscript reports Windows 11, 16 GB RAM and NVIDIA GeForce RTX 4070 Ti (12 GB). Those are paper-reported conditions, not verified runtime requirements. No MATLAB runtime or pretrained network add-ons were available for testing this refactor in the preparation environment. See `docs/VALIDATION.md`.

## 4. Installation and usage

1. Extract this archive, open MATLAB in the repository root (do not recursively add `original_upload` to the path).
2. Install the needed products and model add-ons using MATLAB Add-On Explorer. Check `ver` and optionally `gpuDevice`.
3. Run the lightweight unit checks (no real research images or model weights required):

```matlab
addpath('tests');
run_tests();
```

4. To verify a recovered historical manuscript manifest and all image paths:

```matlab
validate_paper_manifest('data/experiment_manifest.csv', '/absolute/path/to/images');
```

This validates counts and grouping, not the identity of the historical images. Compare filenames and checksums with the experiment owner's records.

5. To launch a **new** supplemental CNN baseline using a verified manifest and fold IDs:

```matlab
cfg = config();
cfg.executionEnvironment = "auto"; % or "cpu" or "gpu"
run_all('data/experiment_manifest.csv', '/absolute/path/to/images', ...
    'results/cnn_run_001', cfg);
```

Use a fresh output directory for each execution. If no historical folds are available and a **new** split is intended, omit the `fold` column and explicitly set `cfg.generateNewFolds = true`. The generated assignments are stored in `effective_manifest.csv`. They are never a substitute for historical folds. To run a new AlexNet experiment, append `"alexnet"` to `cfg.models` before running. The resulting scores cannot be represented as the paper's 70.10% result unless independently established by the authors and manuscript appropriately revised.

6. To verify the published confusion-matrix arithmetic without any dataset or model:

```matlab
m = audit_published_metrics();
```

This reproduces only arithmetic from the 5-by-5 counts printed in manuscript Table 4: 2,626/2,790 correct (~94.12%) and macro-F1 ~94.11%. It does not recover predictions, train the model, establish the 94.10% Table 2 value, or estimate AUC.

## 5. Code information

| Path | Role |
| --- | --- |
| `config.m` | Explicit settings for new CNN experiments |
| `run_all.m` | New grouped-CV training, prediction and logging |
| `validate_paper_manifest.m` | Paper-specific counts, file and group checks |
| `audit_published_metrics.m` | Recomputes Table 4 arithmetic |
| `src/load_dataset.m` | Strict manifest loading and validation |
| `src/assign_grouped_folds.m`, `validate_folds.m` | Group-level fold generation and leakage checks |
| `src/train_cnn.m`, `train_*.m` | Shared transfer-learning implementation and wrappers |
| `src/evaluate_model.m` | Deterministic inference and score class alignment |
| `src/metrics_from_confusion.m`, `save_metrics.m` | Metrics and confusion matrices |
| `src/plot_roc_comparison.m` | One-vs-rest score-based macro ROC and AUC for new runs |
| `tests/run_tests.m` | Lightweight tests including synthetic fixtures created at runtime |
| `original_upload/` | Unmodified source files; not the active implementation |
| `docs/` | Audit, validation status, and manuscript-specific completion checklist |

## 6. Methodology and reproducibility

For new CNN experiments only: stratify five folds at source-image level (or consume supplied folds), train on four folds, predict the held-out fold, and repeat. The evaluation images are never passed into training as validation data. Resizing and grayscale-to-RGB conversion are deterministic; optional training-only horizontal reflection and rotations are configurable and off by default. No ESRGAN or colour jitter is implemented in this refactor. Defaults are Adam, five epochs, minibatch 32, learning rate 0.001, seed 42; these are explicit new baseline settings, not verified historical BA-ViT settings. Each fold uses a fresh pretrained network and new classification head. No hyperparameter optimization on held-out folds is performed.

Report pooled out-of-fold metrics plus mean and sample SD of fold accuracies. The package also reports original-only metrics, which reduce the influence of correlated derivatives. Metrics are fractions in CSV files, not percentages. Rows of confusion matrices are actual labels and columns are predictions. A class with zero metric denominator gets zero by convention. ROC uses actual false-positive-rate coordinates and explicitly aligned score columns; macro-AUC is the arithmetic mean of per-class AUCs. Pooled scores come from different fold models, so AUC is descriptive of the pooled OOF predictions, not an independent external test. MATLAB/GPU operations may be nondeterministic even with fixed random seeds; record versions and hardware, and do not claim bitwise reproducibility.

## 7. Outputs and expected results

`results/cnn_run_001/` will contain configuration and runtime metadata, effective manifest, fold metrics, pooled model summary, per-model predictions and score-column map, fold training information, optional model weights, confusion matrices, class metrics, original-only results, and ROC plots/AUC CSV. These are generated on execution, not supplied as historical results. No fixed accuracy is promised for the new CNN experiments. The separate `results/paper_table_audit/` contains arithmetic-only CSVs calculated from Table 4, with provenance noted in the validation file.

## 8. Citation, licence and contributions

Dataset citation (as stated in the paper): Hussain, T. (2026). *Sugarcane-PK v1.0: A Scale-Referenced Dataset and MATLAB Implementation for Precision Phenotyping of Pakistani Sugarcane Varieties*. Version V1. Zenodo. DOI: 10.5281/zenodo.18619387.

Manuscript citation: Hussain, T., Alharbi, S., Bano, M., and Tan, K. G. *Bud-Based Sugarcane Variety Identification Using a Bud Attention Vision Transformer and Hybrid ACO–SSA Feature Selection*. Submitted manuscript, PeerJ Computer Science, ID 152775. Do not cite it as published without confirming final publication details.

The manuscript declares CC BY 4.0 for the dataset; this does not automatically license source code, pretrained models or third-party software. No code redistribution licence has been granted or inferred by this package. See `LICENSE_STATUS.md`. Authors should approve a suitable code licence before public deposit. MATLAB and pretrained models are obtained from MathWorks under their respective terms and are not redistributed here.

For contributions, document changes, include tests, disclose new experiments, and obtain maintainer approval before release. The corresponding author is the manuscript contact listed in the submitted PDF. Authorship and contributor roles for the actual code release must be confirmed by the authors.
