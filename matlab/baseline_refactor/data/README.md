# Dataset manifest specification

Copy `experiment_manifest_template.csv` to `experiment_manifest.csv` and populate using actual research records. No sample rows are presented as real data.

| Column | Meaning |
| --- | --- |
| image_id | Unique string identifier for each image. Preserve leading zeros. |
| relative_path | Exact path relative to dataset root, including filename and extension. |
| variety | Verified class name. |
| source_image_id | Unique parent original image ID; the same across all derivatives. |
| is_original | 1 for the source photograph, 0 for a derivative. Exactly one original per group. |
| fold | Integer 1–5 from the historical source-grouped assignment, or from a documented new run. Omit the entire column only when explicitly generating new folds. |

Optional additional columns: original_archive_path, image_sha256, plant_id, stalk_id, acquisition_session, transformation, transformation_config, acquisition_site. The loader retains extra columns in the effective manifest but only enforces grouping by source_image_id, not plant or session. The original IDs and mappings cannot be inferred reliably from class names or filenames alone. Do not use fabricated identifiers. This repository does not supply the actual images or assignments.
