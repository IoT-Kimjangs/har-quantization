# WISDM 저장 모델을 단계별로 확인하기

이번에는 WISDM 모델 하나만 준비합니다. 모델 파일 확인 → 실제 로드와 입출력 형태 확인 → 예측 확인 순서로 진행합니다. 각 단계의 결과를 보고 다음으로 넘어갑니다.

## 1. 모델 파일과 커스텀 레이어 준비

현재 로컬에는 `models/wisdm_transformer_best_fp32.keras`를 준비했습니다. 이 파일은 Git에서 제외되므로 저장소를 내려받은 팀원은 별도로 준비해야 합니다.

- 출처: [저자 공개 저장소](https://github.com/AASIFRASHIDKHANDAY/PATCH-BASED-TRANSFORMER-WITH-QUANTIZATION)
- 기준 커밋: `90eb53deec3ff166fa0f1e08211d8d9d1770411d`
- 원본 경로: `models/wisdm_transformer_best_fp32.keras`
- 파일 크기: 2,533,793바이트
- 저장 당시 Keras 버전: 3.13.2
- SHA-256: `c4b50d393409c73e1d259d7f21485e7a9c0e9f9d6f2c43cdd6f4cefb06d654a6`

출처와 검증 기준은 [설정 파일](../configs/wisdm-model.json)에 기록했습니다. 원본 저장소에서 별도의 라이선스 파일을 찾지 못했으므로 모델과 노트북 전체를 이 저장소에 올리지 않습니다. 배포·재사용 조건은 별도로 확인해야 합니다.

다른 팀원이 같은 파일을 준비할 때는 프로젝트 루트에서 다음 명령을 실행합니다. `mktemp`는 원본을 내려받을 새 임시 디렉터리를 만듭니다.

```bash
author_dir=$(mktemp -d /tmp/har-author-models.XXXXXX)
git clone https://github.com/AASIFRASHIDKHANDAY/PATCH-BASED-TRANSFORMER-WITH-QUANTIZATION.git "$author_dir"
git -C "$author_dir" checkout 90eb53deec3ff166fa0f1e08211d8d9d1770411d
mkdir -p models
cp -n "$author_dir/models/wisdm_transformer_best_fp32.keras" models/
.venv/bin/python scripts/inspect_wisdm_model.py
```

검증 명령은 파일 해시, 크기, 저장 버전과 커스텀 레이어 설정을 확인합니다. 모델을 실제로 불러오거나 예측하지 않습니다. `status`가 `archive_checked_only`이고 `weights_loaded`, `prediction_executed`가 모두 `false`이면 이 단계만 마친 것입니다.

### 커스텀 레이어는 왜 필요한가?

원본 모델은 `AddPositionEmbedding`이라는 레이어를 사용합니다. 센서 구간의 순서 정보를 표현에 더하는 역할입니다. 저장된 파일을 읽을 때 Keras가 이 레이어의 동작을 알아야 하므로 [호환 클래스](../src/har_quantization/layers.py)를 준비했습니다.

참고한 정의는 원본 저장소의 `code/Baseline Transformer/WISDMTRANSFORMER_BASELINE.ipynb`에 있습니다. 저장 가중치와 연결되도록 내부 임베딩 이름 `pos_emb`를 유지했습니다. 설정의 `max_len=2048`은 위치 임베딩의 최대 용량이며, 실제 모델에 넣는 입력 길이를 뜻하지 않습니다.

파일 검증 후 실제 모델 로드와 입출력 형태 확인까지 마쳤습니다. 내부 임베딩 가중치를 명시적으로 준비하는 `build()`를 추가해 커스텀 레이어의 미구현 경고를 해결했습니다.

## 2. 직접 모델을 불러와 입출력 형태 확인

Mac ARM64 / Python 3.12.14 / Keras 3.13.2에서 확인했습니다. 같은 결과를 직접 확인하려면 Mac 환경 구성과 의존성 설치를 마친 상태에서 아래 명령을 실행합니다. 환경 구성 변경은 PR #4를 참고합니다.

```bash
cd /Users/jinjeong/IoT-Kimjangs
.venv/bin/python scripts/show_wisdm_shapes.py
```

다른 팀원은 첫 줄을 자신의 프로젝트 경로로 바꿉니다. 이 스크립트는 파일 검증 후 모델을 로드하고 `입력 형태`, `출력 형태`를 출력합니다. 컴파일과 학습, 예측은 실행하지 않습니다.

- 입력 형태는 `(샘플 수, 시간 시점 수, 센서 채널 수)`로 읽습니다.
- 출력 형태는 `(샘플 수, 행동 클래스 수)`로 읽습니다.
- `None`은 해당 차원이 고정되지 않았다는 뜻입니다. 배치 차원에 있으면 한 번에 넣는 샘플 수를 바꿀 수 있습니다.

확인한 결과는 다음과 같습니다.

```text
입력 형태: (None, 80, 3)
출력 형태: (None, 6)
```

샘플 하나는 시간 순서로 나열한 80개 시점과 시점마다 센서 값 3개로 구성됩니다. 샘플 하나를 넣는 배열 형태는 `(1, 80, 3)`이고, 출력은 `(1, 6)`입니다. 80은 시점 수이며 80초를 뜻하지 않습니다. 실제 시간 길이는 데이터 샘플링 주기를 확인해야 합니다. 이 결과는 모델 로드 성공을 의미하며 정확도나 예측 성공을 검증한 결과는 아닙니다.

### 입력의 3개 값은 무엇인가?

센서 장치 3개가 아니라 **스마트폰 가속도 센서의 X·Y·Z축 값 3개**입니다. 원본 코드의 `FEATURES = ["x", "y", "z"]` 순서를 그대로 따라야 합니다. 축은 폰 기준이므로 폰을 어떻게 들고 있는지에 따라 실제 공간에서의 방향은 달라집니다.

```text
샘플 하나: (80, 3)
시점 1:  [X축 가속도, Y축 가속도, Z축 가속도]
시점 2:  [X축 가속도, Y축 가속도, Z축 가속도]
...
시점 80: [X축 가속도, Y축 가속도, Z축 가속도]
```

이 샘플 하나를 모델에 넣을 때는 앞에 배치 차원을 붙여 `(1, 80, 3)`으로 만듭니다. 사람 번호, 정답 행동, 타임스탬프는 입력 채널에 넣지 않습니다. 모델은 80개 시점 동안 세 축의 값이 변하는 패턴으로 행동을 구분합니다.

### 출력의 행동 6개와 순서

아래는 원본 코드의 `LABELS`와 `label2idx`에 지정된 순서입니다. 출력 크기만으로 알아낸 순서가 아니며, 우리도 같은 매핑을 사용해야 합니다.

| 출력 인덱스 | 원본 이름 | 의미 |
| --- | --- | --- |
| 0 | Downstairs | 계단 내려가기 |
| 1 | Jogging | 조깅 |
| 2 | Sitting | 앉아 있기 |
| 3 | Standing | 서 있기 |
| 4 | Upstairs | 계단 올라가기 |
| 5 | Walking | 걷기 |

원본 모델의 마지막 층은 `softmax`입니다. 샘플마다 위 순서로 확률값 6개가 나오고, 합은 반올림 오차를 제외하면 1입니다. 가장 큰 값의 인덱스를 예측 행동으로 읽습니다.

다음은 **설명을 위한 가상 출력이며 실제 예측 결과가 아닙니다.**

```text
[0.02, 0.03, 0.01, 0.02, 0.07, 0.85]
                                   ↑
                         인덱스 5 = Walking
```

이 예시는 걷기로 예측했다는 뜻입니다. `0.85`는 해당 샘플에 대한 모델의 예측값이며 모델 정확도가 85%라는 뜻은 아닙니다. 정확도는 정답이 있는 여러 샘플의 예측을 비교해 따로 측정합니다.

### 형태만 맞추면 바로 예측할 수 있는가?

원본 코드는 `StandardScaler`를 학습 데이터에 맞춘 뒤 학습·평가 입력에 적용합니다. 각 축을 학습 데이터의 평균과 표준편차를 기준으로 바꾸므로, 원시 가속도 배열의 형태만 맞춰 넣어서는 학습 때와 같은 조건이 되지 않습니다.

- 원본 구간 길이는 80개 시점, 이동 간격은 40개 시점입니다. 인접 구간은 50% 겹칩니다.
- X·Y·Z 순서와 학습 때의 표준화 기준을 유지해야 합니다.
- 평가 데이터만으로 표준화 기준을 새로 구하면 원본 조건과 달라집니다.
- 실제 예측 전에 학습 때 사용한 표준화 통계의 확보·재현 방법을 확인해야 합니다. 현재 준비한 모델 파일만으로 그 통계까지 확보했다고 볼 수는 없습니다.

현재 완료 범위는 모델 로드와 입출력 형태 확인입니다. 실제 예측·정확도 평가는 아직 하지 않았습니다.

### 설명의 근거

- [저자 WISDM 기준 노트북: 고정 커밋](https://github.com/AASIFRASHIDKHANDAY/PATCH-BASED-TRANSFORMER-WITH-QUANTIZATION/blob/90eb53deec3ff166fa0f1e08211d8d9d1770411d/code/Baseline%20Transformer/WISDMTRANSFORMER_BASELINE.ipynb): `LABELS`, `label2idx`, `FEATURES`, `WIN`, `STEP`, `StandardScaler`, 마지막 `softmax` 층을 확인했습니다.
- [WISDM 공식 데이터 설명](https://www.cis.fordham.edu/wisdm/dataset.php): 사용 데이터는 Activity Prediction의 `WISDM_ar_v1.1`이며 Actitracker 데이터와 구분합니다. 위 출력 인덱스는 공식 페이지의 나열 순서가 아니라 저자 코드 기준입니다.

## 3. 다음 단계

입출력 형태와 클래스 매핑을 확인했습니다. 다음에는 원본 표준화 통계와 평가 데이터 준비 방법을 확인하고, 조건을 맞춘 시험 입력으로 한 번 예측합니다. 그 다음 실제 평가 데이터, TFLite FP32 변환, 양자화 순서로 진행합니다. 아직 이 단계는 실행하지 않았습니다.
