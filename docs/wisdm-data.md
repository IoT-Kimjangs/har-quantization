# WISDM 데이터 준비

## 현재 준비된 데이터

WISDM 공식 사이트의 **Activity Prediction v1.1**을 받았습니다. 저자 WISDM 기준 코드가 읽는 `WISDM_ar_v1.1_raw.txt`를 사용합니다. Actitracker나 이미 특징을 계산한 ARFF 파일을 입력 데이터로 사용하지 않습니다.

프로젝트 루트 기준 위치는 다음과 같습니다.

```text
data/raw/wisdm/
├── WISDM_ar_latest.tar.gz
└── WISDM_ar_v1.1/
    ├── readme.txt
    ├── WISDM_ar_v1.1_raw.txt
    ├── WISDM_ar_v1.1_raw_about.txt
    ├── WISDM_ar_v1.1_transformed.arff
    └── WISDM_ar_v1.1_trans_about.txt
```

압축 파일과 설명 파일을 함께 보관합니다. `data/raw/`는 Git에서 제외되므로 팀원은 각자 다운로드해야 합니다. 저장소에는 출처·해시·검증 코드와 집계 결과만 올립니다.

- [공식 배포 페이지](https://www.cis.fordham.edu/wisdm/dataset.php)의 Activity Prediction → Download Latest Version
- [공식 다운로드 주소](https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz)
- [파일별 SHA-256과 입력·라벨 순서](../configs/wisdm-data.json)
- [실제 검사 결과](../results/data/wisdm-raw-summary.json)

공식 README는 데이터 사용 시 Kwapisz, Weiss, Moore (2010), *Activity Recognition using Cell Phone Accelerometers*를 인용하고, 데이터를 공유·재배포할 때 `readme.txt`를 포함하도록 요청합니다. 원본 README를 로컬 데이터와 함께 유지합니다.

## 팀원이 같은 데이터를 준비하는 방법

프로젝트 루트에서 실행합니다. Python 표준 라이브러리만 사용하므로 TensorFlow 설치 없이도 가능합니다.

```bash
python3 scripts/prepare_wisdm_data.py --download
```

압축 파일이 없을 때만 공식 서버에서 다운로드합니다. 압축 파일과 내부 파일 해시를 검증한 뒤 정해진 파일만 풉니다. 기존 파일의 내용이 다르면 덮어쓰지 않고 멈춥니다. `latest` 배포 내용이 변경되어도 기존 해시를 자동 갱신하지 않습니다.

터미널에서 서버에 연결할 수 없다면 브라우저로 공식 다운로드 링크를 열고, 받은 `WISDM_ar_latest.tar.gz`를 `data/raw/wisdm/`에 복사한 뒤 실행합니다.

```bash
python3 scripts/prepare_wisdm_data.py
```

이번 Mac에서는 브라우저로 공식 파일을 다운로드한 뒤 검증했습니다. `.env.example`의 `WISDM_DIR=data/raw/wisdm`과 같은 위치입니다. 원본 텍스트는 그 안의 `WISDM_ar_v1.1/` 하위 폴더에 있습니다.

## 파일을 어떻게 읽는가?

원본 한 줄의 구성은 다음과 같습니다. 행 끝의 세미콜론을 제거한 뒤 쉼표로 나눕니다.

```text
사용자 번호,행동,타임스탬프,X축 가속도,Y축 가속도,Z축 가속도;
```

모델 입력 채널은 X·Y·Z 순서입니다. 사용자 번호와 타임스탬프, 정답 행동은 모델 입력 채널이 아닙니다. 행동 인덱스는 저자 코드의 순서를 따릅니다.

| 인덱스 | 행동 |
| --- | --- |
| 0 | Downstairs — 계단 내려가기 |
| 1 | Jogging — 조깅 |
| 2 | Sitting — 앉아 있기 |
| 3 | Standing — 서 있기 |
| 4 | Upstairs — 계단 올라가기 |
| 5 | Walking — 걷기 |

원본 `raw_about.txt`의 명시 샘플링 주기는 **20Hz, 50ms마다 1회**입니다. 입력 80개 시점은 명목상 약 4초 분량입니다. 실제 타임스탬프에 이상이나 간격 차이가 없는지는 구간을 만들 때 별도로 확인해야 합니다.

## 원본 검사 결과

공식 안내의 예제 수는 1,098,207개입니다. 물리적인 줄 수나 검사 조건을 통과한 줄 수와는 구분합니다. 현재 받은 파일에서 한 줄을 한 레코드로 검사한 결과는 다음과 같습니다.

| 항목 | 줄 수 |
| --- | ---: |
| 전체 물리적 줄 | 1,098,210 |
| 빈 줄 | 6 |
| 필드가 6개가 아닌 줄 | 11,738 |
| 숫자로 해석할 수 없는 줄 | 1 |
| NaN·무한대 값이 있는 줄 | 0 |
| 타임스탬프가 0 이하인 줄 | 12,842 |
| 알려지지 않은 행동 | 0 |
| 위 검사 조건을 통과한 줄 | 1,073,623 |

위 사유는 표 순서대로 검사하며 한 줄을 중복 집계하지 않습니다. 형식이 맞지 않는 줄 중에는 여러 레코드가 붙어 있는 줄과 추가 필드가 있는 줄이 있습니다. 저자 코드처럼 한 줄씩 읽어 필드 수가 6개인 경우만 검사했으며, 붙어 있는 레코드를 임의로 복구하지 않았습니다. 숫자는 유한한 값인지도 확인했습니다. 이 기록은 정제된 데이터 파일을 만들었다는 뜻이 아닙니다.

검사 조건을 통과한 데이터는 사용자 36명, 행동 6종입니다.

| 행동 | 줄 수 |
| --- | ---: |
| Downstairs | 100,192 |
| Jogging | 324,600 |
| Sitting | 59,939 |
| Standing | 48,393 |
| Upstairs | 122,598 |
| Walking | 417,901 |

## 여기까지 한 일과 다음 단계

Windows에서 원본 다운로드·해시 검증·형식 검사를 마치고, 저자 공개 코드의 WISDM 전처리 절차로 윈도우를 만들었습니다. 총 26,786개 윈도우를 얻었고 80:20 윈도우 단위 분할 결과는 학습 21,428개, 테스트 5,358개입니다. 전처리 배열은 로컬 `data/processed/`에 두며 원본·전처리 배열은 Git에 올리지 않습니다. 집계·해시는 [데이터 원본 검사 결과](../results/data/wisdm-raw-summary.json)와 [Windows 평가 결과](../results/2026-11-10-windows-wisdm-fp32/metrics.json)에 있습니다.

원본 형식 검사를 통과한 줄만 사용했습니다. 모델 성능 평가는 끝냈지만 저장 모델 학습 당시 scaler 통계 파일은 확보하지 못했습니다. 또한 윈도우 단위 분할에서는 인접 구간이 원본 센서 샘플을 공유하고 같은 사용자가 두 partition에 모두 포함됩니다. 이 결과를 사용자 독립 평가로 해석하지 않으며, 추후 별도 사용자 단위 split 결과와 구분해야 합니다. 검사 통과는 원본 타임스탬프의 연속성을 보증하지 않습니다.

전처리와 라벨의 근거는 [저자 WISDM 기준 노트북의 고정 커밋](https://github.com/AASIFRASHIDKHANDAY/PATCH-BASED-TRANSFORMER-WITH-QUANTIZATION/blob/90eb53deec3ff166fa0f1e08211d8d9d1770411d/code/Baseline%20Transformer/WISDMTRANSFORMER_BASELINE.ipynb)입니다.

새 환경에서 Windows WISDM 전처리와 저장 모델 평가는 다음 순서로 실행합니다. 첫 번째 명령이 로컬 WISDM 원본을 준비하고, 저장된 Keras 모델은 [모델 안내](wisdm-model.md)의 방법으로 먼저 내려받습니다.

```powershell
.venv\Scripts\python.exe scripts\prepare_wisdm_preprocessing.py
.venv\Scripts\python.exe scripts\evaluate_wisdm.py --output results\2026-11-10-windows-wisdm-fp32
```

평가 결과는 [Windows FP32 metrics](../results/2026-11-10-windows-wisdm-fp32/metrics.json)에 있습니다. TFLite 변환·양자화와 성능 측정은 별도 변경으로 제출합니다. See [the Windows TFLite quantization report](windows-tflite-quantization.md) for conversion results; latency and memory measurements follow separately.
