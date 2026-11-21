# Validation record

## Performed during repository preparation

- Inspected the six supplied MATLAB files; preserved byte-identical original copies.
- Reviewed the refactored code statically for manifest integrity, fold isolation, class ordering, metric definitions, and output paths.
- Independently recalculated the manuscript Table 4 confusion counts using Python numerical arithmetic: total 2,790; correct 2,626; accuracy 0.941218637993; macro-F1 0.941094421917. Derived CSVs are included under `results/paper_table_audit/` and are not model-generated predictions.
- Included lightweight MATLAB tests using synthetic image files, fold leakage checks, metric identities, and the paper arithmetic audit.

## Not performed

No MATLAB or Octave executable was available in the preparation environment. MATLAB syntax/runtime compatibility, add-on layer graphs, model training, GPU execution, and the included unit tests have not been executed. No research dataset or checkpoints were supplied. Do not represent static review as successful MATLAB execution. Before release, run `addpath('tests'); run_tests();` in MATLAB R2024b, then execute a small new CNN run with real approved images and execute the recovered historical workflow. Record any fixes and exact output checksums.
