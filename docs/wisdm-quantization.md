# WISDM 양자화와 전체 평가

## 비교 결과

2026-11-10 Mac ARM64 CPU에서 같은 Keras 모델을 FP16·동적 INT8·전체 정수 INT8 설정으로 변환하고, 동일한 평가 구간 **5,358개 전부**를 실행했습니다.

| 모델 | 파일 크기(KiB) | FP32 대비 크기 감소 | 정확도 | Macro-F1 | 정답 수 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TFLite FP32 기준 | 296.167969 | — | 96.771183% | 95.458161% | 5,185 |
| FP16 | 165.773438 | 44.027223% | 96.752520% | 95.439627% | 5,184 |
| 동적 INT8 | 111.210938 | 62.450046% | 96.789847% | 95.473859% | 5,186 |
| INT8 입출력·일부 FP32 연산 | 117.695313 | 60.260621% | 96.715192% | 95.368001% | 5,182 |

1 KiB는 1,024바이트입니다. 크기 감소 기준은 같은 TFLite 런타임의 FP32 모델입니다. Keras 저장 파일과의 형식 차이를 양자화 효과에 포함하지 않았습니다.

FP16은 기준보다 1개 적게, 동적 INT8은 1개 많이, INT8 입출력 모델은 3개 적게 맞혔습니다. 동적 INT8의 작은 정확도 차이를 일반적인 성능 향상으로 단정하지 않습니다. 현재 데이터와 실행 조건의 관측값입니다.

## 전체 정수 INT8 설정의 실제 결과

변환에는 `Optimize.DEFAULT`, `TFLITE_BUILTINS_INT8`, int8 입출력과 학습 대표 데이터를 지정했습니다. SELECT_TF_OPS fallback이나 그래프 구조 변경은 하지 않았습니다.

**변환과 추론은 성공했지만 정수 연산만 남은 모델은 아니었습니다.** 텐서와 연결된 연산을 검사한 결과 다음이 확인됐습니다.

- 입출력은 모두 int8이고, 내부 텐서는 int8 139개·int32 16개·float32 8개입니다.
- LayerNormalization 네 곳에 `DEQUANTIZE → NEG → QUANTIZE` 경로가 남았습니다. `NEG` 4개가 float32 입력·출력을 사용합니다.
- 결과 JSON에 `integer_only_achieved: false`와 해당 연산의 텐서 번호를 기록했습니다.

따라서 코드·파일의 `int8_full`은 **요청한 변환 설정의 식별자**입니다. 비교표와 보고서에서는 실제 산출물을 “INT8 입출력·일부 FP32 연산”으로 부릅니다. 이름과 입출력 dtype만으로 완전 정수 모델이라고 판단하지 않습니다. 연산 진단은 LiteRT 2.3.0의 `_get_ops_details()`를 사용했고 delegate 래퍼를 제외한 모델 연산을 확인했습니다.

## 전체 INT8 보정 데이터

학습 배열 `X_train_s.npy`에서 `np.random.RandomState(42).choice(..., replace=False)`로 **300개 구간**을 고정 선택했습니다. 선택한 학습 배열 인덱스와 전체 구간 번호를 [calibration.csv](../results/2026-11-10-macos-wisdm-quantization/calibration.csv)에 기록했습니다.

보정 구간 번호와 평가 구간 번호의 중복은 0개입니다. 다만 기존 구간 단위 분할 자체에 인접 구간의 센서 시점 중첩이 있으므로, 원본 시점까지 완전히 분리됐다는 뜻은 아닙니다.

INT8 입력은 `round(x / scale + zero_point)` 뒤 int8 범위로 제한하고, 출력은 `(q - zero_point) × scale`로 되돌렸습니다.

| 항목 | 값 |
| --- | ---: |
| 입력 scale | 0.03142688050866127 |
| 입력 zero point | 5 |
| 출력 scale | 0.00390625 |
| 출력 zero point | -128 |
| 평가 입력 전체 값 | 1,285,920 |
| int8 범위를 벗어나 제한된 입력 값 | 59 |

FP16과 동적 INT8 모델의 입출력은 float32입니다. FP16은 가중치 저장 정밀도를 낮춘 방식이며 모든 연산이 FP16으로 실행됐다는 의미는 아닙니다.

## 실행·검증 기록

- Python 3.12.14 / TensorFlow 2.21.0 / Keras 3.13.2 / LiteRT 2.3.0
- 동일한 표준화 평가 입력, 동일한 행동 인덱스 순서, LiteRT CPU 1스레드
- 모델·전처리·기준 FP32 파일과 예측 CSV 해시를 확인한 뒤 평가
- 모든 샘플 출력의 유한성·확률 범위·합 확인. INT8 합 오차 허용 범위에는 출력 양자화 간격 반영
- 정확도·Macro-F1을 혼동행렬로 독립 계산하고, 저장한 전체 CSV에서도 재계산해 일치 확인
- 모델 재학습, 속도·메모리 측정은 수행하지 않음

```bash
.venv/bin/python scripts/quantize_evaluate_wisdm.py --output results/2026-11-10-macos-wisdm-quantization
```

기존 양자화 모델이나 결과가 있으면 덮어쓰지 않습니다. 모델은 로컬 `models/wisdm_transformer_fp16.tflite`, `wisdm_transformer_int8_dynamic.tflite`, `wisdm_transformer_int8_full.tflite`에 저장하며 Git에서 제외합니다.

- [실행 코드](../scripts/quantize_evaluate_wisdm.py)
- [모델별 지표·환경·해시·혼동행렬·정수/실수 텐서 진단](../results/2026-11-10-macos-wisdm-quantization/metrics.json)
- [5,358개 구간별 정답·모델별 예측](../results/2026-11-10-macos-wisdm-quantization/predictions.csv)
- [FP32 기준 변환·평가](wisdm-tflite-fp32.md)

## 다음 작업

Mac에서 Keras 두 호출 방식, TFLite FP32, FP16, 동적 INT8, INT8 입출력 모델의 지연시간·프로세스 피크 메모리를 같은 조건으로 비교합니다. 현재 결과는 파일 크기와 분류 성능 비교이며, 작은 파일이 실제로 더 빠른지는 아직 측정하지 않았습니다.
