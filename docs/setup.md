# 환경 구성

Mac ARM64와 Windows x86-64에서 CPU 실행 조건을 맞추기 위한 초기 환경입니다. 먼저 설치와 작은 모델의 변환·추론을 점검한 뒤 저자 HAR 모델을 불러옵니다.

## 현재 상태

2026-11-03 Mac ARM64에서 Python 3.12.14 기반 `.venv` 설치와 CPU 실행 점검을 완료했습니다. TensorFlow 2.21.0, Keras 3.13.2, LiteRT 2.3.0을 사용해 작은 모델의 Keras 출력과 TFLite FP32 변환 후 LiteRT 추론 출력이 일치함을 확인했습니다. `pip check`도 통과했습니다. **Windows 설치와 실행은 아직 미검증**이며 두 플랫폼 공통 환경은 확정 전입니다.

| 구성 | 버전 후보 | 선택 근거 |
| --- | --- | --- |
| Python | 3.12, 현재 Mac 3.12.14 | TensorFlow·LiteRT의 Mac ARM64 및 Windows x86-64 배포 파일 확인 |
| TensorFlow | 2.21.0 | 두 플랫폼의 Python 3.12 배포 파일 확인 |
| Keras | 3.13.2 | 저자 저장 모델 두 개의 metadata.json에 기록된 버전 |
| LiteRT | ai-edge-litert 2.3.0 | 두 플랫폼의 Python 3.12 배포 파일 확인 |

확인한 저자 저장소 커밋은 `90eb53deec3ff166fa0f1e08211d8d9d1770411d`입니다. 저자 노트북에서 TensorFlow/Keras, NumPy, pandas, scikit-learn, matplotlib, LiteRT 사용을 확인했습니다. psutil과 openpyxl은 후속 메모리 측정과 결과 파일 처리에 사용합니다.

`requirements.txt`는 핵심 세 패키지를 고정하고 보조 패키지는 호환 범위를 지정합니다. 전체 의존성을 고정한 lock 파일이 아닙니다. 두 OS의 실제 실행을 확인한 뒤 공통 버전을 확정합니다.

Mac에서 검증한 패키지 55개의 설치 버전은 [Mac 패키지 목록](../requirements/macos-arm64-py312.txt), 주요 버전과 성공 상태는 [환경 점검 기록](../results/environment/macos-arm64-2026-11-03.json)에 남겼습니다. 패키지 목록은 해당 Mac 환경의 `pip freeze` 기록이며 Windows용 잠금 파일로 간주하지 않습니다. 저자 HAR 모델 로드나 실제 데이터 정확도 검증은 다음 단계입니다.

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

## Windows에서 실행하기

Windows에서는 Mac의 `.venv`나 `requirements/macos-arm64-py312.txt`를 복사해 실행하지 않습니다. **공통 실험 기준은 64비트 Python 3.12와 requirements.txt**입니다. Windows에 Python 3.11이 설치돼 있어도 지우지 않고 3.12를 추가해 별도 가상환경을 만들 수 있습니다.

### 1. Python 버전 확인

PowerShell에서 확인합니다.

```powershell
py -0p
py -3.12 --version
```

3.12가 없으면 [Python 공식 Windows 다운로드](https://www.python.org/downloads/windows/)에서 64비트 Python 3.12를 설치하고 Python launcher를 포함합니다. 설치 후 새 PowerShell에서 위 명령을 다시 실행합니다. OS·CPU가 같아도 Python 3.11 환경은 이번 3.12 환경의 재현 확인으로 취급하지 않습니다.

### 2. PR #4 브랜치 받기

아직 PR이 main에 머지되기 전이라면 main만 받아서는 설치 스크립트가 없습니다. 저장소 루트에서 실행합니다. 기존 작업이 있으면 먼저 보관합니다.

```powershell
git fetch origin
git switch chore/macos-environment
git pull --ff-only origin chore/macos-environment
```

### 3. Windows 전용 설치 명령 실행

기존 `.venv`가 Python 3.11일 수 있으므로, 아래 예시는 별도 폴더를 사용합니다. 기존 환경을 삭제하지 않습니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -VenvPath .venv-py312
```

실행 정책 옵션은 이 PowerShell 프로세스에만 적용됩니다. 가상환경 활성화 명령은 필요하지 않습니다. 스크립트가 64비트 Python 3.12 여부를 확인하고, requirements.txt 설치 → pip check → 작은 모델의 Keras 실행·TFLite 변환·LiteRT CPU 추론 비교를 진행합니다.

스크립트 없이 수동으로 확인하려면 다음과 같습니다.

```powershell
py -3.12 -m venv .venv-py312
.venv-py312\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements.txt
.venv-py312\Scripts\python.exe -m pip check
.venv-py312\Scripts\python.exe scripts/check_environment.py --output .venv-py312/environment.json
```

### 4. 결과 확인

- `.venv-py312/environment.json`의 `status: passed`와 종료 코드 0을 확인합니다.
- 스크립트로 설치하면 `.venv-py312/requirements-windows.txt`에도 설치 버전이 저장됩니다.
- 실패하면 오류 메시지와 위 JSON이 있으면 함께 확인합니다. `py -0p` 결과에서 개인 경로는 가리고 공유합니다.
- 버전 확인에서 멈췄다면 기존 3.11 환경에 억지로 설치하지 말고 Python 3.12 설치 여부와 선택한 가상환경을 확인합니다.
- `DLL load failed` 같은 런타임 오류가 있으면 [TensorFlow 공식 Windows 설치 안내](https://www.tensorflow.org/install/pip#windows-native)의 Visual C++ 런타임 요구 사항도 확인합니다. 이 오류가 실제 발생했다고 가정하고 패키지 버전을 임의 변경하지 않습니다.

## Windows 자동 검증 범위

`Windows environment` 워크플로는 GitHub의 Windows x86-64/Python 3.12 환경에서 실제 Windows 설치 스크립트를 실행합니다. 로그와 `windows-cpu-python312` 아티팩트로 점검 결과를 확인할 수 있습니다. 설치 관련 변경이 있는 PR에 실행되며 필수 머지 조건은 아닙니다.

이 검사는 GitHub Windows 실행 환경의 결과입니다. 팀원 PC에서 같은 명령을 실행한 결과와 구분합니다. Mac ARM64와 Windows x86-64는 하드웨어가 다르므로 패키지·실행 절차와 출력 비교를 맞추는 것이며, OS나 절대 실행 속도가 같아야 한다는 의미는 아닙니다.

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
- 저자 모델의 `AddPositionEmbedding` 등 커스텀 레이어를 확인해 저장 모델을 불러오는 작업은 다음 단계입니다.
- 계획서의 `bench_tflite_variants.py`는 현재 저장소에 포함되지 않았으므로 별도 구현이 필요합니다.

가상환경·캐시·전체 패키지 목록·개인 경로 설정은 `.venv` 또는 `.env`에 보관하며 Git에서 제외합니다. 두 OS 설치 성공 후 공개할 실행 환경 요약과 공통 버전 확정 내역을 PR에 남깁니다.

## 공식 자료

- [TensorFlow 설치 안내](https://www.tensorflow.org/install/pip)
- [TensorFlow 2.21.0 배포 파일](https://pypi.org/project/tensorflow/2.21.0/#files)
- [Keras 3.13.2](https://pypi.org/project/keras/3.13.2/)
- [LiteRT 2.3.0 배포 파일](https://pypi.org/project/ai-edge-litert/2.3.0/#files)
- [저자 저장소](https://github.com/AASIFRASHIDKHANDAY/PATCH-BASED-TRANSFORMER-WITH-QUANTIZATION)
