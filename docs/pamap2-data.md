# PAMAP2 Windows data and model reproduction

## Official data source

The data is PAMAP2 Physical Activity Monitoring from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring), dataset 231, DOI [10.24432/C5NW2H](https://doi.org/10.24432/C5NW2H). UCI lists the dataset under CC BY 4.0. Retain the source readme and cite Reiss and Stricker (2012) when presenting results.

The downloaded outer ZIP is 688,226,940 bytes (SHA-256 `76b3580bd5a804121f507717cb498c66a312ef5c774f4a78ac470e6558add352`). It contains an inner `PAMAP2_Dataset.zip` and `readme.pdf`; the inner archive contains `Protocol/`, `Optional/` and the bundled descriptions. Raw archives and extracted `.dat` files belong in ignored `data/raw/` and are not committed.

Download the outer archive, then verify its pinned size and SHA-256 and safely extract both ZIP layers:

```powershell
New-Item -ItemType Directory -Force data\raw\pamap2 | Out-Null
curl.exe --location --fail --output data\raw\pamap2\pamap2_official.zip https://archive.ics.uci.edu/static/public/231/pamap2%2Bphysical%2Bactivity%2Bmonitoring.zip
.venv\Scripts\python.exe scripts\prepare_pamap2_data.py
```

The extraction script saves a raw-file summary to `results/data/pamap2-raw-summary.json` and extracts the inner archive to the expected `data/raw/pamap2/PAMAP2_Dataset/` path.

## Preprocessing and stored model

The stored model comes from the pinned author repository revision `90eb53deec3ff166fa0f1e08211d8d9d1770411d`; its local path, byte count and SHA-256 are in [the model config](../configs/pamap2-model.json). It takes `(batch, 200, 19)` input and returns 18 class scores. The `.keras` model is excluded from Git because the source repository does not provide a separate model license file.

With the project virtual environment installed and the data extracted, run:

```powershell
.venv\Scripts\python.exe scripts\prepare_pamap2_preprocessing.py
.venv\Scripts\python.exe scripts\evaluate_pamap2.py
```

The preprocessing follows the author's PAMAP2 baseline notebook: recursively list and sort all `.dat` files, select the specified 19 columns, concatenate them, remove activity 0 and rows with nonfinite/missing values, create 200-step windows at stride 100 with majority labels, then make a stratified random 80:20 window split with seed 42 and fit `StandardScaler` on training windows. Both `Protocol/` and `Optional/` are included because the author notebook's recursive glob includes both.

This exact reproduction has material limits. Concatenating files before window creation can create windows across file or participant boundaries. The random window split also allows overlapping samples and participants in both partitions. The saved model file and weights are available, but the original StandardScaler mean and scale used for its training inputs were not. The evaluation therefore refits scaling from this run's training split and cannot certify that it matches the stored model's original input preprocessing. No training was performed.

The official archive check is summarized in [PAMAP2 raw data metadata](../results/data/pamap2-raw-summary.json). The saved-model evaluation metrics and predictions are in [the Windows FP32 results](../results/2026-11-10-windows-pamap2-fp32/). Its `preprocessing-summary.json` records source-file, split, evaluation-array and prediction hashes, plus the run-fitted scaler mean, variance and scale. The run-fitted scaler does not establish the original model training statistics. TFLite conversion and quantization results are documented in [the Windows report](windows-tflite-quantization.md); benchmarks are included in this change.
