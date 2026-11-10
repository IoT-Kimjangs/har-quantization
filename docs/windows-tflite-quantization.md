# Windows TFLite conversion and quantization results

This change converts the available WISDM and PAMAP2 saved Keras models to TFLite, evaluates FP32 and quantized artifacts, and records reproducible metrics. Neither model is retrained. Latency and memory measurements are documented in the [WISDM benchmark report](../results/2026-11-10-windows-wisdm-benchmark-final/windows-results.md) and [PAMAP2 benchmark report](../results/2026-11-10-windows-pamap2-benchmark-final/pamap2-windows-results.md).

## Environment and limits

Results were produced on Windows 11 AMD64, Python 3.12.10, TensorFlow 2.21.0, Keras 3.13.2, LiteRT 2.3.0, NumPy 2.5.4 and scikit-learn 1.9.1. See the [environment record](../results/environment/windows-amd64-2026-11-10.json) and metrics JSON files for artifact hashes and details.

The saved model weights are available, but the original training-time StandardScaler statistics and exact split indices were not. Evaluation refits scaling on this run's training split, so it cannot establish exact reproduction of the original preprocessing. No model training was performed. Integer I/O conversion does not guarantee every operation is integer-only; the WISDM model retains floating-point operators.

## WISDM

FP32 conversion matched Keras predictions on all 5,358 test windows (100% agreement; accuracy 96.771%, macro-F1 95.458%). Its TFLite file is 296.168 KiB.

| Variant | File size | Accuracy | Macro-F1 | Prediction agreement with FP32 |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 296.168 KiB | 96.771% | 95.458% | 100.000% |
| FP16 | 165.773 KiB | 96.753% | 95.440% | 99.981% |
| Dynamic INT8 | 111.211 KiB | 96.790% | 95.474% | 99.925% |
| INT8 I/O | 117.695 KiB | 96.715% | 95.368% | 99.683% |

The INT8 I/O model still contains floating-point operations and is not fully integer-only. See the [FP32 conversion metrics](../results/2026-11-10-windows-wisdm-tflite-fp32/metrics.json), [quantization metrics](../results/2026-11-10-windows-wisdm-quantization/metrics.json), calibration sample record and predictions in those result folders.

## PAMAP2

Evaluation used 494 test windows. TFLite FP32 is the comparison baseline.

| Variant | File size | Accuracy | Macro-F1 | Prediction agreement with FP32 |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 996.961 KiB | 97.368% | 96.736% | 100.000% |
| FP16 | 523.469 KiB | 97.368% | 96.736% | 100.000% |
| Dynamic INT8 | 315.875 KiB | 97.368% | 96.736% | 100.000% |
| INT8 I/O | 326.680 KiB | 97.571% | 96.867% | 99.798% |

See the [PAMAP2 quantization metrics](../results/2026-11-10-windows-pamap2-quantization/metrics.json), calibration sample record and predictions in that result folder. The INT8 I/O model retains floating-point operations (`DEQUANTIZE`, `NEG`, and `QUANTIZE`) and is not fully integer-only.

## Reproduce

Run from the repository root in the prepared Windows virtual environment, after following the WISDM and PAMAP2 data/model setup instructions:

```powershell
.venv\Scripts\python.exe scripts/convert_evaluate_wisdm_fp32.py --output results/2026-11-10-windows-wisdm-tflite-fp32
.venv\Scripts\python.exe scripts/quantize_evaluate_wisdm.py --output results/2026-11-10-windows-wisdm-quantization --fp32-result results/2026-11-10-windows-wisdm-tflite-fp32
.venv\Scripts\python.exe scripts/quantize_evaluate_pamap2.py
```

These commands are for a fresh run. The scripts refuse to overwrite existing outputs; move or remove an existing output directory and model artifact before rerunning. Scripts write model binaries under ignored `models/` and metrics, calibration records and predictions under `results/`. Model binaries are not committed.
