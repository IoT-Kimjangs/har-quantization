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

여기까지 파일 검증은 완료했습니다. 클래스 문법은 확인했지만 실제 모델 로드 호환성은 2번에서 확인합니다.

## 2. 직접 모델을 불러와 입출력 형태 확인

이 단계는 아직 실행하지 않았습니다. Mac 환경 구성과 의존성 설치를 마친 상태에서 아래 명령을 직접 실행합니다. 환경 구성 변경은 PR #4를 참고합니다.

```bash
cd /Users/jinjeong/IoT-Kimjangs
.venv/bin/python scripts/show_wisdm_shapes.py
```

다른 팀원은 첫 줄을 자신의 프로젝트 경로로 바꿉니다. 이 스크립트는 파일 검증 후 모델을 로드하고 `입력 형태`, `출력 형태`를 출력합니다. 컴파일과 학습, 예측은 실행하지 않습니다.

- 입력 형태는 `(샘플 수, 시간 시점 수, 센서 채널 수)`로 읽습니다.
- 출력 형태는 `(샘플 수, 행동 클래스 수)`로 읽습니다.
- `None`은 해당 차원이 고정되지 않았다는 뜻입니다. 배치 차원에 있으면 한 번에 넣는 샘플 수를 바꿀 수 있습니다.

출력된 숫자를 확인하고 우리 데이터의 구간 길이와 채널 구성이 맞는지 함께 해석합니다. 오류가 나면 오류 내용을 확인한 뒤 진행합니다. 숫자만으로 센서 축의 순서나 행동 이름까지 알 수는 없으므로 원본 전처리와 라벨 매핑도 확인해야 합니다.

## 3. 다음 단계

입출력 형태를 확인한 뒤, 원본 전처리와 클래스 매핑을 확인하고 시험 입력으로 한 번 예측합니다. 그 다음 실제 평가 데이터, TFLite FP32 변환, 양자화 순서로 진행합니다. 아직 이 단계는 실행하지 않았습니다.
