# Bud-Based Sugarcane Variety Identification Using a Bud Attention Vision Transformer and Hybrid ACO–SSA Feature Selection

PeerJ Computer Science | Manuscript 152775 | AI Application | Reproducibility package, author-review release

Manuscript authors: Tassadaq Hussain, Soltan Alharbi, Mukhtiar Bano, Kim Geok Tan.

## 1. Description and provenance

This package accompanies a study of sugarcane bud image classification. It preserves the six supplied original MATLAB CNN baseline scripts and includes corrected MATLAB CNN workflows, a newly written Python BA-ViT-style reference and hybrid ACO–SSA search, and a MATLAB downstream reference consuming fold-specific BA-ViT features exported by Python. **The added implementations are new reference implementations, not the original historical source of the manuscript's 94.10% accuracy.** No historical experiment has been rerun. Do not claim exact reproduction or submit these as the original experimental code. See `docs/PEERJ_COMPLIANCE.md` and `docs/REPRODUCIBILITY_STATUS.md`.

## 2. Dataset information

Hussain, T. (2026). Sugarcane-PK v1.0: A Scale-Referenced Dataset and MATLAB Implementation for Precision Phenotyping of Pakistani Sugarcane Varieties. Zenodo. Published 12 February 2026. Version V1. DOI: https://doi.org/10.5281/zenodo.18619387. Dataset license per author-supplied Zenodo record: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The author-supplied record identifies 7,362 RGB images in eight morphological categories and the file `Sugarcane_Varieties_Dataset.zip` (667.6 MB; MD5 2ea3cdc0ba74710e154f6353f5700519). The live archive was not downloaded or inspected here. The manuscript reports 1,000 original bud photographs and 1,790 derivatives; the parent archive alone does not identify those exact 2,790 records.

| Variety | Originals | Working set |
|---|---:|---:|
| CP77-400 | 200 | 563 |
| CPF-237 | 200 | 560 |
| CPF-250 | 200 | 550 |
| CPF-251 | 200 | 553 |
| CPF-253 | 200 | 564 |

No images/checkpoints are bundled. Download and extract the parent archive using Python 3.11+:

```sh
python scripts/fetch_dataset.py --download --extract
```

The tool checks the record checksum and archive MD5; its network behavior has not been tested here. If needed use the DOI landing page to download manually. Inspect the archive metadata and obtain the exact experiment subset, derivatives and fold mappings from the original experiment owner. Use `data/experiment_manifest_template.csv` to record inputs. Each row is one image; paths are relative to `--root`; every derivative shares its original's `source_image_id`; original rows have identical `image_id` and `source_image_id`; `is_original` is 0/1. Fold IDs 1–5 must be supplied by the historical experiment owner or generated explicitly for new experiments only. The SHA-256, archive path, source URL and license fields document provenance and should be filled by the authors. Do not invent missing mappings. Refer to the MATLAB baseline data guide for additional schema details.

## 3. Software requirements

Python 3.11+ (Python 3.11 runtime family recommended), PyTorch, torchvision, numpy, pandas, Pillow, scikit-learn, pytest. Install `requirements.txt` in a virtual environment; choose compatible PyTorch CPU or CUDA wheels. MATLAB R2024b, Statistics and Machine Learning Toolbox, Deep Learning Toolbox and Image Processing Toolbox; CNN model support packages for ResNet-50, MobileNetV2 and optional AlexNet. Optional Parallel Computing Toolbox for GPU training. The original manuscript reports Windows 11, 16 GB RAM, RTX 4070 Ti 12 GB. Training this new reference may have different resource requirements.

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q tests/test_reference.py
```

## 4. Code information and usage

| Path | Purpose |
|---|---|
| `original_upload/` | Exact six supplied original baseline MATLAB files; not active code |
| `matlab/baseline_refactor/` | Corrected CNN baselines, optional AlexNet, grouped CV, ROC and arithmetic audit |
| `python/reference.py` | Manifest validation, BA-ViT-style network, ACO–SSA and classifiers |
| `python/run_reference.py` | New end-to-end grouped cross-validation and fold exports |
| `python/enhance_reference.py` | Optional external TorchScript ESRGAN-compatible model adapter |
| `matlab/reference/` | New MATLAB ACO–SSA and SVM/KNN evaluation using exported features |
| `configs/reference.json` | Explicit new reference settings |
| `scripts/fetch_dataset.py` | Parent dataset download and checksum validation |
| `tests/` | Synthetic checks of grouping, model dimensions, selection and metrics |
| `docs/` | Compliance, permissions, limitations and provenance |

Launch Python from the repository root with an author-prepared manifest:

```sh
python python/run_reference.py --manifest data/experiment_manifest.csv --root /path/to/images --output results/new_run --config configs/reference.json
```

For a **new** experiment only, add `--new-folds` to create a seed-controlled stratified grouping; add `--alexnet` for the optional new pretrained AlexNet comparison (requires weights download). The program saves effective manifest, fold checkpoints, feature CSVs, masks, fitness histories, predictions and pooled/fold metrics. Class indices follow the table order, starting at zero. Reuse a fresh output directory for every run. The coarse KNN is skipped if fewer than 100 training rows. `--new-folds` does not recover historical folds. GPU deterministic operations are requested with warnings; exact cross-device repeatability is not guaranteed.

Optional enhancement (only with an author-supplied compatible model):

```sh
python python/enhance_reference.py --model /path/to/compatible_model.pt --input /path/to/original.png --output /path/to/derived.png --scale 4
```

This adapter does not provide ESRGAN weights, train ESRGAN or reproduce the historical enhancement recipe. Record derivatives and parent IDs in the manifest before training. The scale above is an example, not a historical setting.

MATLAB downstream reference (after Python produces fold exports):

```matlab
addpath('matlab/reference');
run_feature_reference('results/new_run','results/matlab_new_run');
```

The MATLAB implementation runs a different stochastic search and classifier configuration on the same fold-specific features; it is not an independent MATLAB BA-ViT training implementation or an exact Python numerical port. For corrected original CNN baselines, `cd matlab/baseline_refactor` in MATLAB and follow its README. Run `audit_published_metrics()` there to check the manuscript confusion matrix independently of model training.

## 5. Methodology and settings

New Python reference: input 384×384 RGB, 16×16 patches (576 tokens), 8 attention heads, global token pooling, 1,024-D output. The embedding width (192), depth (4), attention residual, initialization from scratch, augmentation probabilities, five training epochs and AdamW are newly chosen settings. Historical pretrained weights, exact MATLAB graph and internode pretraining stage were not supplied; this reference does not reproduce that stage. Train each fold on its training groups only, use deterministic resizing for held-out evaluation, and fit feature selection within training groups with an inner grouped holdout. Because the extractor is fitted to the complete outer training fold, the inner search uses embeddings learned partly from inner validation images; this is a tuning heuristic, not an independent inner performance estimate. Outer evaluation remains untouched by fitting. The 10-member population, 100 iterations, evaporation 0.2 and feature counts 100, 250, 500, 750, 1000 follow reported settings; the fitness weights 0.99 and 0.01, fixed-cardinality selection, ACO update and SSA ranking are new choices. At fixed cardinality the size penalty is constant. Classifier configurations approximate the reported families, not exact historical MATLAB presets. The working-set metrics include correlated derivatives; plant-level independence is not established. 

## 6. Expected results and validation

The paper *reports* 94.10% accuracy for cubic SVM with 1,000 selected features, 70.10% AlexNet accuracy and 94.12% pooled confusion-matrix accuracy. These are historical manuscript claims, not expected results guaranteed by this code. The bundled MATLAB arithmetic audit recomputes only published counts. Four Python reference unit tests passed on synthetic fixtures. Full training, download, MATLAB execution and historical metric reproduction were not performed. See `docs/VALIDATION.md`.

## 7. Citation, license and contributions

Cite the Zenodo dataset using the DOI above and the companion manuscript by its authors once a publication DOI is assigned. `CITATION.cff` records a draft citation without inventing a DOI. The dataset is reported as CC BY 4.0; verify the included Zenodo LICENSE scope for original code. Software redistribution license approval is pending; see `LICENSE_STATUS.md`. Contribution instructions are in `CONTRIBUTING.md`. The paper authors must review all reference code and approve the release. The repository explicitly distinguishes supplied and newly written code; disclose AI-assisted code preparation as required by the journal.
