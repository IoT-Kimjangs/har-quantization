# Runtime and quantization effects across Windows and Mac

This analysis separates three effects: Keras API call path (`predict()` to direct model call), runtime conversion (Keras direct call to TFLite FP32), and TFLite quantization (FP32 to FP16 or INT8). Values are medians in milliseconds under each platform's default thread settings, transcribed from the platform reports.

## WISDM

| Path | Windows (ms) | Mac (ms) | Within-platform change |
| --- | ---: | ---: | --- |
| Keras `predict()` | 208.7349 | 24.4097 | Reference |
| Keras direct call | 107.8605 | 14.2137 | 48.3% / 41.8% lower than `predict()` |
| TFLite FP32 | 0.0488 | 0.0448 | Runtime conversion effect; not quantization |
| TFLite FP16 | 0.0399 | 0.0451 | 18.2% lower / 0.8% higher than FP32 |
| TFLite dynamic INT8 | 0.0333 | 0.0309 | 31.8% / 31.0% lower than FP32 |
| TFLite INT8 I/O, mixed FP32 ops | 0.1921 | 0.0693 | 293.6% / 54.8% higher than FP32 |

The Keras-to-TFLite FP32 transition is much larger than the quantization changes in these measurements. It combines runtime, invocation, and tensor handling differences; it cannot be attributed to quantization. Dynamic INT8 had the lowest median among the TFLite variants on both sessions. FP16 latency was close to FP32, and INT8 I/O with remaining FP32 operations was slower.

## PAMAP2

| Path | Windows (ms) | Mac (ms) | Within-platform change |
| --- | ---: | ---: | --- |
| Keras `predict()` | 213.0406 | 25.6873 | Reference |
| Keras direct call | 165.5518 | 19.6295 | 22.3% / 23.6% lower than `predict()` |
| TFLite FP32 | 0.5795 | 0.2554 | Runtime conversion effect; not quantization |
| TFLite FP16 | 0.1566 | 0.2559 | 73.0% lower / 0.2% higher than FP32 |
| TFLite dynamic INT8 | 0.1132 | 0.1184 | 80.5% / 53.6% lower than FP32 |
| TFLite INT8 I/O, mixed FP32 ops | 1.8892 | 0.2629 | 226.0% / 2.9% higher than FP32 |

Within each platform, dynamic INT8 had the lowest median among TFLite variants. The Windows PAMAP2 benchmark sampled with replacement from 494 windows (420 unique); Mac cycled through all 494 windows. Therefore these cross-platform timing numbers are descriptive and do not isolate an OS or hardware effect.

## Interpretation

- The three comparisons answer different questions. `predict()` versus direct call measures Keras API path overhead; direct Keras versus TFLite FP32 includes a runtime change; only TFLite FP32 versus quantized variants isolates the quantization configuration within a runtime.
- These are single-session observations from different CPUs and system states. Background load, CPU frequency, and thermal conditions were not controlled. Do not interpret cross-platform differences as a causal OS effect.
- WISDM used 1,000 non-replacement samples from 5,358 test windows on both platforms. PAMAP2 used different input schedules as described above.
- Accuracy and Macro-F1 are full test-set metrics in the source reports, not metrics calculated from the timed calls. PAMAP2's window-level split is not subject-independent; its original training scaler and exact split artifacts were unavailable.
- Peak RSS includes runtime, model, and inputs. It is not a measure of model weights alone.
- INT8 I/O variants retain floating-point operations and are not fully integer-only models.

## Source reports

- [Windows WISDM](https://github.com/IoT-Kimjangs/har-quantization/blob/exp/windows-complete/results/2026-11-10-windows-wisdm-benchmark-final/windows-results.md)
- [Windows PAMAP2](https://github.com/IoT-Kimjangs/har-quantization/blob/exp/windows-complete/results/2026-11-10-windows-pamap2-benchmark-final/pamap2-windows-results.md)
- [Mac WISDM](https://github.com/IoT-Kimjangs/har-quantization/blob/exp/wisdm-macos-benchmark/docs/wisdm-macos-benchmark.md)
- [Mac PAMAP2](https://github.com/IoT-Kimjangs/har-quantization/blob/exp/pamap2-macos-pipeline/docs/pamap2-macos-results.md)
