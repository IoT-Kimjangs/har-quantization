# WISDM TFLite FP32 변환·전체 평가

## 결과

2026-11-09 Mac ARM64 CPU에서 공개 Keras 모델을 양자화 없이 TFLite FP32로 변환했습니다. 같은 평가 데이터 **5,358개 전체**를 Keras와 LiteRT에 입력해 비교했습니다.

| 항목 | Keras | TFLite FP32 |
| --- | ---: | ---: |
| 정확도 | 96.771183% | 96.771183% |
| Macro-F1 | 95.458161% | 95.458161% |
| 정답 구간 | 5,185 | 5,185 |
| 파일 크기 | 2,474.407227 KiB | 296.167969 KiB |

예측 행동은 **5,358개 전부 일치**, 불일치는 0개입니다. 혼동행렬도 동일합니다. 출력 확률의 최대 절대 차이는 `1.0728836059570312e-6`, 평균 절대 차이는 `8.931190680065563e-9`였습니다. 확률의 미세한 수치 차이가 예측 행동을 바꾸지는 않았습니다.

파일 크기 감소에는 Keras 저장 형식에서 TFLite 추론 형식으로 바뀐 효과가 포함됩니다. FP16·INT8 양자화 효과로 해석하지 않습니다. 1 KiB는 1,024바이트이며, 실제 TFLite 크기는 **303,276바이트**입니다.

## 변환·평가 조건

- Python 3.12.14, TensorFlow 2.21.0, Keras 3.13.2, LiteRT 2.3.0
- 원본 Keras 모델을 `compile=False`로 로드하고 SavedModel로 임시 내보내기
- `TFLiteConverter.from_saved_model()` 사용, `optimizations=[]`
- `TFLITE_BUILTINS`만 허용, SELECT_TF_OPS나 커스텀 연산 fallback을 추가하지 않음
- 입출력: `(1, 80, 3)` → `(1, 6)`, 모두 `float32`
- 텐서 목록: float32 141개, 인덱스 등에 쓰는 int32 8개. FP16·INT8 텐서 없음
- LiteRT CPU, 스레드 1, 로그에서 기본 XNNPACK CPU delegate 생성 확인
- Keras는 배치 256으로 `training=False`, LiteRT는 배치 1로 각 평가 구간 실행
- 모델·전처리 산출물의 SHA-256 검증 후 같은 입력·정답 순서로 전체 평가

모든 출력 확률의 유한성·범위·합을 확인했습니다. 정확도·Macro-F1은 혼동행렬로 별도 계산한 결과와도 대조했습니다.

## 실행 및 기록

Mac 환경·원본 모델·WISDM 전처리 산출물을 준비한 뒤 프로젝트 루트에서 실행합니다.

```bash
.venv/bin/python scripts/convert_evaluate_wisdm_fp32.py --output results/2026-11-09-macos-wisdm-tflite-fp32
```

기존 모델이나 결과가 있으면 덮어쓰지 않고 멈춥니다. 모델은 로컬 `models/wisdm_transformer_fp32.tflite`에 보관하며 Git에서 제외합니다. 임시 SavedModel은 변환 후 정리합니다.

- [실행 코드](../scripts/convert_evaluate_wisdm_fp32.py)
- [전체 지표·혼동행렬·환경·입출력·파일 해시](../results/2026-11-09-macos-wisdm-tflite-fp32/metrics.json)
- [평가 구간별 Keras·TFLite 예측과 출력 오차](../results/2026-11-09-macos-wisdm-tflite-fp32/predictions.csv)
- [이전 Keras 기준 평가](wisdm-evaluation.md)

구간별 CSV에는 센서 원본 없이 구간 번호·정답·두 모델의 예측 인덱스·출력 오차를 기록했습니다. 모델과 원본 데이터는 재배포하지 않습니다.

## 해석과 다음 작업

**현재 평가셋에서 FP32 변환으로 분류 성능이 바뀌지 않았음을 확인했습니다.** 공개 코드로 재계산한 표준화·구간 단위 분할 조건과 기존 재현 한계는 유지됩니다. 원본 학습 산출물의 완전한 동일성을 추가로 증명한 결과는 아닙니다.

지연시간과 메모리는 아직 측정하지 않았으므로 파일이 작아진 만큼 빨라졌다고 결론 내리지 않습니다. 다음은 FP16·동적 INT8·전체 정수 INT8 변환과 같은 평가셋의 분류 성능 확인입니다. 그 뒤 Mac에서 같은 조건으로 속도·메모리를 비교합니다.
