# 환경 구성

Mac과 Windows는 Python·TensorFlow·Keras·LiteRT 버전의 공통 지원 여부를 확인한 뒤 버전을 맞춥니다. 아직 검증하지 않은 버전을 확정된 실행 환경으로 적지 않습니다.

## 개인 환경

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell에서는 `py -m venv .venv`와 `.venv\Scripts\Activate.ps1`을 사용합니다. 모델 로드·변환·추론까지 두 OS에서 확인한 뒤 의존성 파일을 PR로 추가합니다. 현재 저장소에는 실행 가능한 벤치마크 구현이나 검증된 패키지 잠금 파일이 없습니다.

## 데이터와 모델

- WISDM과 PAMAP2는 공식 배포처에서 각자 다운로드하고 이용 조건을 확인합니다.
- 원본은 `data/raw/`, 전처리 결과는 `data/processed/`, 모델은 `models/`에 둡니다.
- 로컬 경로는 `.env.example`을 `.env`로 복사해 적습니다. 후속 실행 코드는 이 설정을 읽도록 구현합니다.
- 저자 노트북에 하드코딩된 Windows 경로는 로컬 설정으로 바꿉니다.
- 저장 모델을 읽을 때 저자 구현의 `AddPositionEmbedding` 등 커스텀 레이어 정의가 필요합니다.
- 계획서에 언급된 `bench_tflite_variants.py`는 제공된 자료에 포함되지 않았으므로 별도 확보 또는 구현이 필요합니다.

환경 기록은 `results/environment-template.json`을 복사해 실행별 폴더에 저장합니다. 결과를 낼 때 `python -m pip freeze` 기록도 남깁니다.
