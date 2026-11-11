# Mac·Windows 통합 분석 초안

서진정의 Mac 결과와 장은성의 Windows 결과를 같은 연산 스레드 모드끼리 묶은 통합 분석입니다. 새 실험이 아니라 이미 제출된 원시 기록을 검증·집계한 결과입니다.

[통합 보고서·표·그래프](../results/platform-integration-draft/report.md)를 먼저 읽으면 됩니다. 전체 숫자는 [48조건 CSV](../results/platform-integration-draft/summary.csv), 플랫폼 내부 개선율은 [40개 비교 CSV](../results/platform-integration-draft/effects.csv)에 있습니다.

## 지금 할 수 있는 일과 남은 일

- 지금: 정확도·Macro-F1·크기·중앙 지연시간·p95·프로세스 피크 메모리를 통합하고 Keras 호출·런타임 변경·양자화 효과를 나눠 해석합니다.
- 선행 PR: [#7 WISDM Mac](https://github.com/IoT-Kimjangs/har-quantization/pull/7), [#14 PAMAP2 Mac](https://github.com/IoT-Kimjangs/har-quantization/pull/14), [#17 Windows](https://github.com/IoT-Kimjangs/har-quantization/pull/17). 2026-11-11(KST) 재검토 기준 세 PR 모두 main에 머지됐습니다. 고정 출처와 main의 원시 실험 파일이 같은지 대조했습니다.
- 선행 PR의 main 반영과 충돌 정리를 완료해 통합 PR은 리뷰 가능한 상태로 전환합니다. 결과 문서는 발표·보고서용 초안이며 최종 제출 전 검토가 필요합니다. 이후 원시 결과가 변경되면 고정 출처와 달라지는지 다시 확인합니다.
- Windows의 별도 비교 PR #18·#19는 통합 비교를 이 작업에서 진행하기로 해 닫았습니다. 그 문서나 수동 전사 숫자는 입력으로 사용하지 않았습니다. #17의 비교표 역시 집계 입력으로 쓰지 않고 실험 JSON/CSV만 사용합니다.
- GitHub main 반영 규칙은 그대로입니다. 이 PR의 리뷰 요청은 장은성에게만 보냅니다.

## 출처 고정

[설정](../configs/platform-result-sources.json)에 전체 커밋 SHA와 경로를 고정했습니다. 작업 브랜치가 삭제돼도 [출처 목록](../results/platform-integration-draft/sources.json)의 커밋 URL을 통해 원본을 추적할 수 있습니다. 원본 데이터·모델 바이너리·기존 실험 CSV는 이 PR에 중복 추가하지 않았습니다.

| 데이터·플랫폼 | PR | 고정 커밋 |
| --- | --- | --- |
| WISDM Mac | #7 | `2f2b6c04089da5bcd96ccaddbe15251e2253e942` |
| PAMAP2 Mac | #14 | `edee959d239bf696832f869f6b523a579a0eead1` |
| WISDM·PAMAP2 Windows | #17 | `cb5d712b2773a4aced0c3893a13141c4687c1c67` |

이후 PR에 새 커밋이 올라와도 자동으로 섞이지 않습니다. 데이터를 바꿀 때는 설정의 고정 커밋을 검토해 갱신하고 새 폴더에 다시 생성합니다.

## 재생성

저장소 루트의 Python 환경에 NumPy와 matplotlib가 필요합니다. 기존 프로젝트 가상환경을 사용할 수 있습니다. Git은 PATH에 있어야 합니다. 아래 fetch는 GitHub에서 고정 커밋을 가져오며 모델·데이터를 다운로드하거나 학습/추론을 실행하지 않습니다.

```bash
# 처음 복제한 저장소에서는 원본 PR 커밋을 먼저 확보합니다.
git fetch origin 2f2b6c04089da5bcd96ccaddbe15251e2253e942 edee959d239bf696832f869f6b523a579a0eead1 cb5d712b2773a4aced0c3893a13141c4687c1c67

# Mac: 빈 새 폴더로 재생성합니다. 기존 산출물은 덮어쓰지 않습니다.
.venv/bin/python scripts/integrate_platform_results.py --output /tmp/har-platform-integration-check
```

Windows에서는 같은 스크립트를 다음처럼 실행할 수 있습니다. 이 통합 스크립트를 실제 검증한 환경은 Mac이며, Windows에서 재실행한 것은 아닙니다.

```powershell
.venv\Scripts\python.exe scripts/integrate_platform_results.py --output results/platform-integration-check
```

기본 출력은 `results/platform-integration-draft/`이며 이미 파일이 있으면 중단합니다. 원본 커밋이 없거나 해시·수치·조건이 다르면 완료 결과를 만들지 않습니다. 출력 파일에 새 측정 일시를 붙이지 않고 기존 실험의 출처를 유지합니다.

## 자동 검증 범위

1. 전체 평가 CSV의 SHA-256과 정확도·Macro-F1·혼동행렬을 대조합니다. 타이밍 표본으로 정확도를 추정하지 않습니다.
2. 48개 측정 조건이 빠짐없이 있는지 확인하고 총 48,000행에서 CSV 해시·반복 번호·양수 지연시간·워밍업 50·배치 1을 검사합니다.
3. 원시 나노초에서 중앙값·선형 보간 p95·평균을 다시 계산해 summary JSON 및 케이스 JSON과 비교합니다.
4. 각 데이터셋·플랫폼의 12조건 입력 순서/구간 번호를 대조하고, 48,000개 예측이 전체 평가의 해당 모델 예측과 일치하는지 확인합니다.
5. 벤치마크 모델 해시와 TFLite 크기를 평가 기록에 대조합니다. Mac·Windows 전체 평가의 입력·정답·분할 배열 해시도 대조합니다.
6. `thread_mode`를 키로 삼아 같은 모드끼리 표·효과·그림을 만듭니다. `default`를 1스레드 값으로 대체하지 않습니다.

이는 기록의 정합성 검사입니다. Windows에서 실행한 과정 자체를 여기서 재실행하거나 실제 장치의 현재 성능을 측정한 것은 아닙니다.

## 해석할 때 지킬 점

- 양자화 효과는 같은 플랫폼·같은 스레드의 **TFLite FP32 대비**로 계산합니다. Keras→TFLite 개선을 양자화 효과로 합치지 않습니다.
- Mac/Windows는 다른 CPU·장치·세션이므로 OS 자체의 인과 효과로 해석하지 않습니다. 같은 스레드 모드라고 런타임 내부 동작 전체가 같아지는 것은 아닙니다.
- PAMAP2 타이밍은 Mac 494개 전체 순환, Windows 복원추출 고유 420개로 서로 다릅니다. 전체 평가 입력 해시가 같다는 사실과 타이밍 입력 순서/빈도가 같다는 주장은 구분합니다.
- macOS native peak RSS와 Windows peak working set은 서로 다른 OS 계측입니다. 원시 단위는 byte이며 표에만 MiB로 변환했습니다. OS 간 메모리 절감률은 계산하지 않습니다.
- 원 학습 scaler·분할 미확인, 피험자·센서 시점 중첩, PAMAP2 파일 경계와 논문 수치 미재현 한계는 유지합니다. INT8 입출력도 일부 FP32 연산이 남아 있습니다.

## 추가 시각 자료

기존 지연시간/p95와 양자화 개선율 그림 외에 다음 세 그림을 같은 재생성 명령으로 만듭니다. 발표자료에서 확대해 쓸 수 있도록 PNG와 SVG를 함께 제공합니다. 그림 안의 영문 표기는 플랫폼별 글꼴 차이로 글자가 깨지는 것을 피하기 위한 것이며, 읽는 방법과 한계는 보고서에 한국어로 적었습니다.

| 그림 | 읽는 방법 | 데이터/범위 |
| --- | --- | --- |
| [크기–추론시간](../results/platform-integration-draft/size-latency.png) | 오른쪽일수록 큰 파일, 위일수록 느린 추론. 색은 변형, 원은 Mac·사각형은 Windows | TFLite 4변형 × 2데이터 × 2플랫폼 × 2스레드 = [32점](../results/platform-integration-draft/size-latency.csv). Keras 런타임 변경 효과는 기존 그림으로 분리 |
| [메모리 변화율](../results/platform-integration-draft/memory-effects.png) | 같은 플랫폼·스레드의 FP32 기준. 양수는 감소, 음수는 증가. 라벨에 대상 피크 MiB도 표시 | 3양자화 변형의 [24개 비교](../results/platform-integration-draft/memory-effects.csv). OS끼리의 메모리 비율이 아님 |
| [실험 흐름도](../results/platform-integration-draft/experiment-workflow.png) | 모델/공식 데이터 → 분할·표준화·보정 → 전체 평가/타이밍 → 기록 통합 | 전체 평가와 벤치마크 입력을 구분하고 원 학습 통계 미확인 한계를 명시 |

SVG 파일은 같은 이름의 `.svg`로 저장됩니다. 모델 크기가 같아 점이 같은 위치에 겹칠 수 있으며, 점마다 별도의 반복 실험을 수행했다는 의미는 아닙니다. 산점도에서 회귀선이나 일반적인 상관관계를 주장하지 않습니다. 프로세스 메모리 차이는 한 번의 실행에서 관측한 작은 변화이므로 유의미한 RAM 절감이 입증됐다고 해석하지 않습니다.
