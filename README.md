# HAR 양자화 효율 분석

초소형 IoT 센서 기반 행동 인식 모델의 양자화 효율을 분석하는 사물인터넷 Project 2 저장소입니다. 모델 파일이 작아진 만큼 실제 추론도 빨라지는지 확인하고, 런타임 변경 효과와 양자화 효과를 나눠 측정합니다.

## 처음 읽는 팀원에게

- [우리 프로젝트 쉽게 이해하기](docs/project-explained.md): 무엇을 하는지, 왜 측정하는지, 각자 맡을 일을 쉬운 말로 정리했습니다.
- [Project 2 수행계획서 Word 사본](docs/project-proposal.docx): 제출 계획과 역할 분담을 확인할 수 있습니다. 공개용 사본에서는 학번을 제외했습니다.

## 연구 질문

- Keras에서 TFLite로 바꿨을 때와 TFLite FP32를 양자화했을 때의 개선 폭은 각각 얼마인가?
- 파일 크기와 압축률이 지연시간·메모리 사용량을 설명하는가?
- Mac(Apple Silicon)과 Windows(x86)에서 양자화 방식별 순위가 유지되는가?

## 실험 범위

| 항목 | 계획 |
| --- | --- |
| 데이터 | WISDM, PAMAP2 |
| 실행 방식 | Keras predict(), Keras model(x), TFLite FP32, FP16, INT8 동적, INT8 전체 정수 |
| 플랫폼 | Mac ARM, Windows x86 |
| 스레드 | 1스레드, 런타임 기본 설정 |
| 측정 조합 | 2 × 6 × 2 × 2 = 48개 |
| 지표 | 파일 크기, 파라미터 수, 지연시간 중앙값·p95, 프로세스 피크 메모리, 정확도, Macro-F1 |

배치 1, 워밍업 50회, 측정 1,000회를 기준으로 합니다. 플랫폼 간 절대 속도만으로 하드웨어 우열을 판단하지 않습니다.

## 시작하기

[Mac·Windows 통합 분석](docs/platform-results-integration.md)에 같은 스레드 조건의 통합표·그래프와 고정 출처를 정리했습니다. 선행 실험 PR #7·#14·#17은 main에 반영됐습니다.

WISDM 저장 모델은 [단계별 모델 확인](docs/wisdm-model.md), PAMAP2는 [데이터·모델 재현 안내](docs/pamap2-data.md)에 따라 준비합니다.

[PAMAP2 Mac 실험 결과](docs/pamap2-macos-results.md)에 데이터·전처리 원본 대조, 논문 수치와의 차이, 변환·양자화·성능 측정을 정리했습니다.

[WISDM Mac 속도·메모리 비교](docs/wisdm-macos-benchmark.md)에 12개 조건의 실측 결과와 원시 기록을 정리했습니다.

[WISDM 양자화 비교](docs/wisdm-quantization.md)에 FP16·동적 INT8·INT8 입출력 모델의 크기·분류 성능과 연산 진단을 정리했습니다.

[TFLite FP32 변환·전체 평가](docs/wisdm-tflite-fp32.md)에서 Keras와 예측 행동 100% 일치를 확인했습니다.

WISDM 전체 평가에서 [정확도·Macro-F1이 논문 표시 정밀도까지 일치](docs/wisdm-evaluation.md)했습니다.

논문과 코드가 맞는지는 [쉬운 대조표와 리뷰 체크리스트](docs/paper-code-check.md)에서 확인합니다.

WISDM 분할·표준화는 [공개 코드 기준 재계산](docs/wisdm-preprocessing.md)에 정리했습니다.

WISDM 원본은 [데이터 안내](docs/wisdm-data.md), PAMAP2는 [데이터·전처리·평가 안내](docs/pamap2-data.md)를 참고합니다. Windows 실행 환경은 [환경 기록](results/environment/windows-amd64-2026-11-10.json)과 [패키지 목록](requirements/windows-amd64-py312.txt)에, 변환·양자화 및 속도·메모리 결과는 [Windows 결과 안내](docs/windows-tflite-quantization.md)에 있습니다.

Windows 저장 모델 평가는 2026-11-10 기록이며 모델을 재학습하지 않았습니다. 원 scaler 확보 및 윈도우 단위 split의 일반화 한계를 보고서에 함께 반영하고, Mac의 같은 조건 결과를 모은 뒤 플랫폼별 분석을 진행합니다.

1. [협업 규칙](CONTRIBUTING.md)을 읽습니다.
2. [환경 구성](docs/setup.md)에 따라 개인 환경과 데이터 경로를 기록합니다.
3. [실험 규약](docs/experiment-protocol.md)에 따라 조건을 맞춥니다.
4. 작업 브랜치에서 수정하고 PR을 올려 리뷰를 받습니다.

```bash
git clone https://github.com/IoT-Kimjangs/har-quantization.git
cd har-quantization
git switch -c docs/environment-notes
```

## 저장소 구성

- `docs/`: 프로젝트 계획, 환경 구성, 실험 규약, GitHub 운영 규칙
- `configs/`: 실험 조건
- `src/har_quantization/`: 재현·변환·측정 코드
- `notebooks/`: 탐색 및 분석 노트북
- `results/`: 검토 가능한 측정 결과와 실행 환경 기록
- `data/`, `models/`: 개인 로컬 데이터와 모델 보관 위치, 대용량 파일은 Git 제외
- `.github/`: PR·이슈 양식, 리뷰 검사, 저장소 설정 명세

## 논문 정리

- [두 논문이 과제에 어떻게 이어지는지와 읽는 순서](docs/papers/README.md)
- [TinyML HAR 서베이 쉬운 정리](docs/papers/tinyml-har-survey.md)
- [양자화 패치 Transformer 구조·결과·한계 정리](docs/papers/quantized-patch-transformer.md)

## 참고 논문과 코드

수행계획서에서 선정한 자료입니다. 출판 정보와 논문 수치는 발표·보고서 인용 전 원문에서 다시 확인합니다.

- 서베이: *A Review of TinyML for Human Activity Recognition on Edge Devices* — [DOI](https://doi.org/10.3390/make8090271)
- 구현체: *Edge efficient human activity recognition using a quantized patch based transformer* — [DOI](https://doi.org/10.1038/s41598-026-62744-4)
- [저자 공개 코드](https://github.com/AASIFRASHIDKHANDAY/PATCH-BASED-TRANSFORMER-WITH-QUANTIZATION)

원본 코드·모델·데이터를 가져오기 전에 각각의 라이선스와 재배포 조건을 확인합니다. 이 저장소는 원본 구현체를 복제해 배포하지 않습니다.

## 일정과 협업

발표는 **12/1(화)**, 발표자료는 **11/30(월) 오전 11:59**, 보고서 초안은 **12/4(금)**, 최종은 **12/14(월)**입니다. 제공된 계획 문서 기준이며 초안·최종 제출 시각은 LMS에서 확인합니다.

- [주차별 할 일·담당·완료 기준](docs/project-plan.md)
- [브랜치 생성부터 PR·승인·머지까지](docs/pr-guide.md)

## 팀과 진행 기록

김가은: 논문 분석·보고서·발표 총괄. 서진정·장은성: 환경 구성·재현·확장 실험을 OS별로 분담합니다. 세부 분담은 [프로젝트 계획](docs/project-plan.md)에 기록합니다.

| 날짜 | 내용 |
| --- | --- |
| 2026-10-27 | 프로젝트 구조, 실험 규약, PR·리뷰 규칙 구성 |
| 2026-11-10 | Windows saved-model evaluation for WISDM and PAMAP2 (see follow-up [TFLite conversion/quantization PR #15](https://github.com/IoT-Kimjangs/har-quantization/pull/15) and [benchmark PR #16](https://github.com/IoT-Kimjangs/har-quantization/pull/16)) |

기존 계획서의 사전 검증 수치는 본 저장소에서 재현한 결과로 취급하지 않습니다. 재현 결과는 실행 조건과 원시 기록을 포함한 PR로 추가합니다.

TFLite conversion and quantization results are documented in [the Windows report](docs/windows-tflite-quantization.md). Benchmark results are available in the [WISDM report](results/2026-11-10-windows-wisdm-benchmark-final/windows-results.md) and [PAMAP2 report](results/2026-11-10-windows-pamap2-benchmark-final/pamap2-windows-results.md), with raw measurements and plots alongside them.
