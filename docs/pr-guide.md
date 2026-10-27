# PR 작성부터 머지까지

PR은 작업 브랜치의 변경을 main에 반영하기 전에 팀원이 검토하는 절차입니다. 문서·실험 결과·코드 모두 PR을 사용합니다. 원격 작업 브랜치에 push하는 것은 가능하지만 main에 직접 push하지 않습니다.

## 1 작업을 이슈로 정리

Issues → New issue에서 작업 또는 실험 양식을 선택합니다. 할 일, 완료 조건, 담당자, 참고 자료를 적고 이슈 번호를 기록합니다. 예를 들어 WISDM 평가 코드와 Windows 환경 문서는 각각 따로 검토할 수 있으므로 별도 PR로 나눕니다.

## 2 최신 main에서 브랜치 만들기

```bash
git switch main
git pull --ff-only origin main
git switch -c feat/wisdm-evaluation
```

브랜치는 feat/기능명, fix/수정내용, docs/문서명, exp/실험명, chore/설정명 형식을 사용합니다. 개인 OS 측정은 `exp/macos-benchmark`, `exp/windows-benchmark`처럼 구분합니다. 이미 올린 PR을 수정할 때는 새 브랜치를 만들지 말고 같은 브랜치에서 작업합니다.

## 3 수정 내용을 확인하고 push

```bash
git status
git diff
python scripts/check_repository.py
node --test .github/scripts/review-policy.test.cjs
git add src/har_quantization/ docs/
git diff --cached
git commit -m "feat: WISDM 저장 모델 평가 추가"
git push -u origin feat/wisdm-evaluation
```

위 add 경로는 예시입니다. 실제 수정한 파일만 지정합니다. 메타데이터 검사는 모델 정확도나 실험 코드의 정상 실행을 보장하지 않으므로, 관련 실행 명령과 결과를 PR에 별도로 적습니다. 설치된 프로그램이 없어 실행하지 못했다면 그 사실과 필요한 확인을 남깁니다.

데이터 원본·모델 바이너리·토큰·.env·학번이 포함된 제출 원본·개인 절대경로는 올리지 않습니다. CSV·그림·실행 메모는 공개 가능한 내용인지 확인합니다.

## 4 GitHub에서 PR 열기

1. 저장소 상단 Compare & pull request를 누르거나 Pull requests → New pull request를 선택합니다.
2. base는 `main`, compare는 본인 작업 브랜치인지 확인합니다.
3. 제목은 실제 변경을 한 문장으로 씁니다. 예: `exp: Windows INT8 지연시간 측정 결과 추가`.
4. 본문에 변경 이유, 변경 내용, 확인 방법·결과, 리뷰가 필요한 부분을 적습니다.
5. 관련 이슈를 해결하는 PR이면 `Closes #이슈번호`를 적습니다. 참고만 하는 이슈는 `Refs #이슈번호`를 씁니다.
6. Assignees에 작성자를, Reviewers에 `.github/reviewers.json`에 등록된 지정 리뷰어를 추가합니다. 본문에 이름만 적는 것은 리뷰 요청이 아닙니다.
7. 미완성은 Create draft pull request로 열고, 검토 준비가 되면 Ready for review로 바꿉니다.

### PR 본문 예시

```markdown
## 변경 이유
Windows에서도 양자화 모델별 지연시간을 같은 조건으로 비교하려고 합니다.

## 변경 내용
- WISDM의 6개 실행 방식에 대한 측정 결과를 추가했습니다.
- 환경 정보와 반복별 지연시간 CSV를 함께 남겼습니다.

## 확인 방법과 결과
- 실행 명령: 실제로 실행한 명령을 적습니다.
- 환경: OS, CPU, 패키지 버전, 스레드, delegate를 적습니다.
- 조건: batch 1, warmup 50, 측정 1,000회 여부를 적습니다.
- 결과: 실제 측정값 또는 결과 파일 경로를 적습니다.
- 미확인 항목: 아직 확인하지 못한 내용이 있으면 적습니다.

## 리뷰 요청
측정 범위와 TFLite FP32 기준 압축률 계산을 확인해주세요.

Closes #이슈번호
```

숫자와 실행 명령은 예시를 그대로 복사하지 말고 본인 작업에서 확인한 값으로 바꿉니다. 문서 PR은 링크·날짜·출처 확인 결과를 적으면 됩니다.

## 5 리뷰어가 확인하고 승인

리뷰어는 Files changed에서 diff를 읽고, 필요한 경우 해당 브랜치를 받아 직접 실행합니다. 데이터 분할·seed·측정 조건·수치 근거·민감정보 포함 여부를 변경에 맞게 확인합니다.

질문은 줄별 댓글로, 반영이 필요한 문제는 Request changes로 남깁니다. 승인할 때는 Review changes → 검토 내용 작성 → Approve → Submit review 순서로 제출합니다. 예: “배치와 반복 횟수, FP32 기준 압축률 계산을 확인했습니다. 결과 CSV와 표의 값이 일치합니다.” 실제 확인한 내용만 적습니다.

일반 댓글만 달거나 Approve 본문을 비워 두면 현재 지정 리뷰어 검사 조건을 충족하지 않습니다. 작성자는 본인 PR을 승인할 수 없습니다.

## 6 피드백 반영과 재리뷰

작성자는 같은 브랜치에서 수정·커밋·push합니다. 기존 PR에 자동으로 반영되므로 새 PR을 만들 필요가 없습니다. 댓글에 어떻게 수정했는지 답하고 검토가 끝난 대화를 해결합니다. 새 커밋을 올리면 이전 승인이 무효가 되므로 지정 리뷰어가 최신 커밋을 다시 승인해야 합니다.

main이 바뀌어 동기화가 필요하면 GitHub의 Update branch를 사용하거나 아래처럼 반영합니다. 이력 재작성은 팀원과 협의한 경우에만 사용합니다.

```bash
git fetch origin
git switch feat/wisdm-evaluation
git merge origin/main
# 충돌이 있으면 파일을 수정하고 git add 후 git commit
git push
```

## 7 머지와 브랜치 정리

지정 리뷰어 전원의 내용 있는 승인, 최소 1명 승인, 마지막 push한 사람 외 승인, 미해결 대화 없음, `repository-check`·`review-approval` 성공, 최신 main과의 동기화를 확인합니다. 충족되지 않으면 우회하지 말고 실패 이유를 해결합니다.

Squash and merge를 선택하고 최종 변경을 설명하는 제목·본문을 확인한 뒤 머지합니다. 원격 작업 브랜치는 자동 삭제됩니다. main으로 이동한 뒤 로컬 브랜치를 정리합니다.

```bash
git switch main
git pull --ff-only origin main
git fetch --prune
git branch -d feat/wisdm-evaluation
```

Squash merge는 로컬 커밋 이력이 main과 달라 `-d`가 거절될 수 있습니다. GitHub의 Merged 표시와 변경 반영, 미푸시 작업이 없는지 확인한 경우에만 `git branch -D feat/wisdm-evaluation`로 해당 로컬 브랜치를 지웁니다.

## 현재 지정 리뷰어와 검사 상태

작성자가 `improvv`이면 `20232541-gaeun`·`20327135`, `20232541-gaeun`이면 `improvv`·`20327135`, `20327135`이면 `improvv`·`20232541-gaeun` 두 명이 지정 리뷰어입니다. Reviewers에 해당 두 명을 요청합니다. 세 명 모두 저장소에서 기여할 수 있는 권한을 갖습니다.

두 리뷰어가 최신 커밋에 검토 내용을 적은 Approve를 제출하기 전에는 `review-approval`이 Pending입니다. 이것은 코드 오류나 실행 실패가 아닙니다. 새 커밋·승인 취소·수정 요청이 있으면 다시 Pending으로 바뀝니다. 실제 워크플로 실행 오류는 별도의 실패로 표시됩니다. 예전 `review-policy` 실패 알림은 과거 실행 기록에 남을 수 있습니다.
