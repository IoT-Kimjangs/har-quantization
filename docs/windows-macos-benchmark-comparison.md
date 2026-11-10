# Windows and Mac inference benchmark comparison

This report compares the Windows results in this PR with the Mac results from PRs [#12](https://github.com/IoT-Kimjangs/har-quantization/pull/12) (WISDM) and [#14](https://github.com/IoT-Kimjangs/har-quantization/pull/14) (PAMAP2). The measurements are observations from one PC and one Mac session, not controlled estimates of an OS effect.

## Measurement conditions

Both reports use batch size 1, 50 warmups, 1,000 timed calls, independent worker processes, and report median and p95 latency plus process peak RSS. Model loading and file I/O are excluded; TFLite input/output handling is included. Neither run controlled background load, CPU frequency, or thermal state.

| Dataset | Windows input selection | Mac input selection | Cross-platform comparability |
| --- | --- | --- | --- |
| WISDM | 1,000 non-replacement samples from 5,358 test windows | 1,000 non-replacement samples from 5,358 test windows | Same population and sample count; timings are still single-session observations. |
| PAMAP2 | 1,000 replacement samples from 494 windows; 420 unique windows | Cycles through all 494 windows to make 1,000 calls | Different input frequency/order; do not interpret timing differences as a direct platform comparison. |

## Median latency, default thread settings

| Dataset | Path | Windows (ms) | Mac (ms) | Interpretation |
| --- | --- | ---: | ---: | --- |
| WISDM | Keras predict | 208.7349 | 24.4097 | Different hardware/runtime sessions; descriptive only. |
| WISDM | Keras direct call | 107.8605 | 14.2137 | Different hardware/runtime sessions; descriptive only. |
| WISDM | TFLite FP32 | 0.0488 | 0.0448 | Descriptive; same sample population and count. |
| WISDM | TFLite FP16 | 0.0399 | 0.0451 | Descriptive; same sample population and count. |
| WISDM | TFLite dynamic INT8 | 0.0333 | 0.0309 | Descriptive; same sample population and count. |
| WISDM | TFLite INT8 I/O, mixed FP32 ops | 0.1921 | 0.0693 | Operator paths differ; descriptive only. |
| PAMAP2 | Keras predict | 213.0406 | 25.9707295 | Different input sampling and hardware/runtime sessions. |
| PAMAP2 | Keras direct call | 165.5518 | 19.391375 | Different input sampling and hardware/runtime sessions. |
| PAMAP2 | TFLite FP32 | 0.5795 | 0.255625 | Sampling differs; not a direct platform comparison. |
| PAMAP2 | TFLite FP16 | 0.1566 | 0.256042 | Sampling differs; not a direct platform comparison. |
| PAMAP2 | TFLite dynamic INT8 | 0.1132 | 0.119292 | Sampling differs; not a direct platform comparison. |
| PAMAP2 | TFLite INT8 I/O, mixed FP32 ops | 1.8892 | 0.262917 | Sampling and operator paths differ; descriptive only. |

## Interpretation limits

Compare model paths within each platform first. Do not claim these measurements isolate a Windows-versus-Mac effect: CPU models, runtimes, delegates, process state, and the sampled input sequence for PAMAP2 differ. Peak RSS includes the Python/runtime process, model, and input arrays; it is not model-weight memory. Accuracy comes from the separate full test-set evaluations, not the 1,000 timing calls.

PAMAP2 remains a window-level random split, not a subject-independent split. Its stored-model evaluation refits the scaler because original training statistics and split indices are unavailable. The INT8 I/O variants retain floating-point operations and are not fully integer-only.

## Source reports

- [Windows WISDM measurements](../results/2026-11-10-windows-wisdm-benchmark-final/windows-results.md)
- [Windows PAMAP2 measurements](../results/2026-11-10-windows-pamap2-benchmark-final/pamap2-windows-results.md)
- [Mac WISDM measurements (PR #12)](https://github.com/IoT-Kimjangs/har-quantization/pull/12)
- [Mac PAMAP2 measurements (PR #14)](https://github.com/IoT-Kimjangs/har-quantization/pull/14)
- [Mac PAMAP2 default-thread summary at commit `edee959`](https://github.com/IoT-Kimjangs/har-quantization/blob/edee959d239bf696832f869f6b523a579a0eead1/results/2026-11-10-macos-pamap2-benchmark/summary.json)
