# 환경 구성

Mac ARM64와 Windows x86-64에서 CPU 실행 조건을 맞추기 위한 초기 환경입니다. 먼저 설치와 작은 모델의 변환·추론을 점검한 뒤 저자 HAR 모델을 불러옵니다.

## 현재 상태

2026-11-03 Mac ARM64에서 Python 3.12.14 기반 `.venv` 설치와 CPU 실행 점검을 완료했습니다. TensorFlow 2.21.0, Keras 3.13.2, LiteRT 2.3.0을 사용해 작은 모델의 Keras 출력과 TFLite FP32 변환 후 LiteRT 추론 출력이 일치함을 확인했습니다. `pip check`도 통과했습니다. Windows 11 AMD64 / Python 3.12.10에서도 CPU 환경 확인, 저장 모델의 WISDM·PAMAP2 평가, TFLite 변환·양자화 및 추론 성능 측정을 수행했습니다([Windows 결과 PR #17](https://github.com/IoT-Kimjangs/har-quantization/pull/17)). 확인 범위는 해당 Windows PC와 실행 조건에 한정됩니다.

| 구성 | 버전 후보 | 선택 근거 |
| --- | --- | --- |
| Python | 3.12, 현재 Mac 3.12.14 | TensorFlow·LiteRT의 Mac ARM64 및 Windows x86-64 배포 파일 확인 |
| TensorFlow | 2.21.0 | 두 플랫폼의 Python 3.12 배포 파일 확인 |
| Keras | 3.13.2 | 저자 저장 모델 두 개의 metadata.json에 기록된 버전 |
| LiteRT | ai-edge-litert 2.3.0 | 두 플랫폼의 Python 3.12 배포 파일 확인 |

확인한 저자 저장소 커밋은 `90eb53deec3ff166fa0f1e08211d8d9d1770411d`입니다. 저자 노트북에서 TensorFlow/Keras, NumPy, pandas, scikit-learn, matplotlib, LiteRT 사용을 확인했습니다. psutil과 openpyxl은 후속 메모리 측정과 결과 파일 처리에 사용합니다.

`requirements.txt`는 핵심 세 패키지를 고정하고 보조 패키지는 호환 범위를 지정합니다. 전체 의존성을 고정한 lock 파일이 아닙니다. Python 3.12와 TensorFlow 2.21.0, Keras 3.13.2, LiteRT 2.3.0 조합은 Mac ARM64와 Windows x86-64의 CPU 실행에서 확인했으며, 이 문서는 모든 기기·환경에 대한 호환성을 보장하지 않습니다.

Mac에서 검증한 패키지 55개의 설치 버전은 [Mac 패키지 목록](../requirements/macos-arm64-py312.txt), 주요 버전과 성공 상태는 [환경 점검 기록](../results/environment/macos-arm64-2026-11-03.json)에 남겼습니다. 패키지 목록은 해당 Mac 환경의 `pip freeze` 기록이며 Windows용 잠금 파일로 간주하지 않습니다. 이 환경 점검 기록 자체는 HAR 평가를 대신하지 않지만, Mac HAR 실험은 이미 수행됐습니다. 결과는 [WISDM 평가](wisdm-evaluation.md), [WISDM 양자화](wisdm-quantization.md), [WISDM 벤치마크](wisdm-macos-benchmark.md), [PAMAP2 Mac 결과](pamap2-macos-results.md)에서 확인할 수 있습니다. Windows 평가 결과와 한계는 [WISDM](../results/2026-11-10-windows-wisdm-benchmark-final/windows-results.md), [PAMAP2](../results/2026-11-10-windows-pamap2-benchmark-final/pamap2-windows-results.md) 보고서를 참조하세요.

## Mac에서 설치

현재 이 Mac에는 `.venv` 설치와 점검이 완료되어 있습니다. 평소 작업을 시작할 때는 `source .venv/bin/activate`로 활성화합니다. 새 Mac에서 설치하거나 기존 환경을 다시 점검하려면 아래 명령을 사용합니다.

```bash
cd /Users/jinjeong/IoT-Kimjangs
bash scripts/setup_macos.sh
source .venv/bin/activate
```

다른 Mac에서 처음 구성할 때는 ARM64 Python 3.12를 먼저 준비합니다. `python3.12`를 찾지 못하면 설치된 실행 파일을 `PYTHON_BIN`으로 지정합니다.

```bash
PYTHON_BIN=/path/to/python3.12 bash scripts/setup_macos.sh
```

스크립트는 프로젝트 전용 가상환경을 사용하고 공식 PyPI에서 의존성을 설치합니다. GPU 플러그인을 추가하지 않으며 점검은 CPU로 실행합니다. 시스템 Python과 전역 패키지는 변경하지 않습니다.

설치 후 스크립트가 자동으로 수행하는 확인은 다음과 같습니다.

1. Python 버전과 ARM64 확인
2. 패키지 설치와 `pip check`
3. 필요한 패키지 import와 핵심 버전 확인
4. 작은 Dense 모델을 Keras에서 실행하고 TFLite FP32로 변환
5. LiteRT CPU 추론 출력과 Keras 출력을 비교
6. `.venv/environment.json`과 `.venv/requirements-macos.txt` 기록

이 점검은 환경 확인용이며 논문 HAR 모델 로드나 실제 데이터셋 정확도 재현을 대신하지 않습니다.

## Windows 실행 결과

Windows 11 AMD64 / Python 3.12.10에서 CPU 환경 확인, WISDM·PAMAP2 저장 모델 평가, TFLite 변환과 양자화, 추론 성능 측정을 수행했습니다. 환경 정보와 패키지 목록, 명령 및 결과는 [Windows 결과 PR #17](https://github.com/IoT-Kimjangs/har-quantization/pull/17)에서 확인할 수 있습니다.

저장 모델을 재학습하지 않았습니다. 원 학습의 scaler 통계와 정확한 분할 인덱스를 확보하지 못해 평가 학습 분할에서 scaler를 다시 적합했습니다. PAMAP2는 윈도우 단위 무작위 분할이며 피험자 독립 평가가 아닙니다. 지표, 전처리 해시, 원시 측정값과 실행 명령은 Windows 결과 자료를 참조하세요.

Windows와 Mac 결과는 각각 한 번의 환경에서 관측한 값입니다. 입력 샘플링 차이와 해석상 한계는 [저장소의 플랫폼 비교 문서](windows-macos-benchmark-comparison.md)에 정리했습니다. 통합표와 양자화 효과 분석은 [PR #22](https://github.com/IoT-Kimjangs/har-quantization/pull/22)에서 확인할 수 있습니다. Windows 원시 결과는 위의 WISDM·PAMAP2 보고서에서 확인할 수 있습니다.

## Windows에서 같은 환경 후보 설치

64비트 Python 3.12를 준비한 뒤 저장소 루트에서 실행합니다. 아래 절차는 새 Windows 환경을 설치하고 초기 점검하는 방법이며, 위에 기록한 결과는 별도의 특정 Windows PC에서 수행했습니다.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements.txt
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe scripts/check_environment.py --output .venv/environment.json
.venv\Scripts\python.exe -m pip freeze > .venv/requirements-windows.txt
```

가상환경 활성화 권한에 상관없이 실행하도록 Python 경로를 직접 지정했습니다. `pip check`와 환경 확인은 패키지·작은 모델 경로를 점검하며, Windows HAR 실험의 전체 결과를 새 PC에서 재현했다는 의미는 아닙니다.

## 다시 점검하기

Mac은 다음 명령을 실행합니다.

```bash
.venv/bin/python -m pip check
.venv/bin/python scripts/check_environment.py --output .venv/environment.json
```

`status: passed`와 종료 코드 0은 작은 Keras 모델의 변환·LiteRT 추론까지 확인됐다는 뜻입니다. `incomplete`는 Python 버전·가상환경·패키지 설치 확인이 남은 상태이며 `failed`는 import·변환·추론 과정에 오류가 난 상태입니다.

`No matching distribution found`가 나오면 앞의 네트워크 오류도 함께 확인합니다. 이번 설치 시도에서는 PyPI 호스트 DNS 조회 실패가 확인됐으므로 패키지 버전 부재로 단정하지 않습니다.

## 데이터와 모델

- WISDM·PAMAP2는 공식 배포처에서 다운로드하고 이용 조건을 확인합니다.
- 원본은 `data/raw/`, 전처리 결과는 `data/processed/`, 모델은 `models/`에 둡니다.
- `.env.example`을 `.env`로 복사해 개인 경로를 기록합니다. 기존 `.env`는 덮어쓰지 않습니다. 현재 점검 스크립트는 데이터 경로를 사용하지 않습니다.
- 저자 노트북의 하드코딩된 Windows 경로는 이후 로컬 설정으로 바꿉니다.
- 저자 모델의 `AddPositionEmbedding` 등 커스텀 레이어는 저장 모델 평가에서 사용했습니다. 이 문서의 Mac 설치 점검은 작은 Dense 모델을 대상으로 하며, Windows HAR 실험 전체를 Mac에서 재현했다는 의미는 아닙니다.
- Windows 측정은 [`benchmark_wisdm_windows.py`](../scripts/benchmark_wisdm_windows.py), [`benchmark_pamap2_windows.py`](../scripts/benchmark_pamap2_windows.py)를 사용하고, Mac 측정 스크립트는 [`benchmark_wisdm_macos.py`](../scripts/benchmark_wisdm_macos.py), [`benchmark_pamap2_macos.py`](../scripts/benchmark_pamap2_macos.py)입니다. 생성된 원시 측정과 요약은 각 플랫폼 결과 안내 문서에서 확인합니다.

가상환경·캐시·전체 패키지 목록·개인 경로 설정은 `.venv` 또는 `.env`에 보관하며 Git에서 제외합니다. 두 OS에서 확인한 실행 환경 요약과 공통 버전 후보의 검증 범위는 관련 PR에 기록합니다.

## 공식 자료

- [TensorFlow 설치 안내](https://www.tensorflow.org/install/pip)
- [TensorFlow 2.21.0 배포 파일](https://pypi.org/project/tensorflow/2.21.0/#files)
- [Keras 3.13.2](https://pypi.org/project/keras/3.13.2/)
- [LiteRT 2.3.0 배포 파일](https://pypi.org/project/ai-edge-litert/2.3.0/#files)
- [저자 저장소](https://github.com/AASIFRASHIDKHANDAY/PATCH-BASED-TRANSFORMER-WITH-QUANTIZATION)
