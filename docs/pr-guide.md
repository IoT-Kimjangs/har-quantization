# PR 작성부터 머지까지

**브랜치에서 작업 → PR 생성 → 작성자 외 팀원 한 명 Approve → Squash merge** 순서입니다. 승인 리뷰의 본문 작성은 선택입니다.

## 1. 작업 브랜치 만들기

```bash
git switch main
git pull --ff-only origin main
git switch -c feat/wisdm-evaluation
```

이미 열린 PR을 수정할 때는 같은 작업 브랜치에 커밋·push합니다. 새 PR을 다시 만들 필요는 없습니다.

## 2. 수정 내용을 확인하고 push

```bash
git status
git diff
python scripts/check_repository.py
git add <수정한-파일>
git diff --cached
git commit -m "feat: WISDM 평가 추가"
git push -u origin feat/wisdm-evaluation
```

설치한 패키지와 관련 실행 명령으로 변경을 확인하고 실제 결과를 기록합니다. 원본 데이터·대용량 모델·토큰·개인정보는 올리지 않습니다.

## 3. GitHub에서 PR 열기

1. Pull requests → New pull request 또는 Compare & pull request를 누릅니다.
2. base는 main, compare는 작업 브랜치로 선택합니다.
3. 제목과 본문에 변경 내용, 이유, 확인 결과를 적습니다.
4. Reviewers에 작성자 외 쓰기 권한이 있는 리뷰어 한 명을 지정합니다. 본문에 이름만 적는 것은 리뷰 요청이 아닙니다.
5. 작업이 끝났으면 Ready for review 상태로 둡니다. 작업 중이면 Draft를 사용합니다.

현재 리뷰 요청 대상은 다음과 같습니다. 후보가 두 명인 경우 그중 한 명을 선택하면 됩니다.

| 작성자 | 리뷰 요청 대상 |
| --- | --- |
| improvv | 20327135만 지정 |
| 20232541-gaeun | improvv 또는 20327135 |
| 20327135 | improvv 또는 20232541-gaeun |

## 4. 리뷰어가 승인하기

Files changed에서 변경을 확인한 뒤 **Review changes → Approve → Submit review**를 누릅니다. 리뷰 본문은 비워도 됩니다. 일반 댓글만 남기면 승인 수에 포함되지 않습니다.

질문이나 수정할 점이 있으면 댓글 또는 Request changes를 사용합니다. 작성자는 필요한 내용을 반영하고 팀원에게 확인을 요청합니다. 새 커밋만으로 이전 승인을 자동 취소하지는 않습니다.

## 5. 머지하기

작성자 외 쓰기 권한이 있는 한 명이 승인하면 **Squash and merge**를 누릅니다. 제목·본문을 확인한 뒤 머지합니다. 원격 작업 브랜치는 자동 삭제됩니다.

최신 main과의 동기화, 승인 본문, 마지막 push 외 추가 승인, 대화 해결, 저장소 검사 통과를 별도 필수 조건으로 요구하지 않습니다. 실제 충돌이 있다면 해결해야 합니다. Draft는 Ready for review로 바꿉니다.

머지 후 로컬은 아래처럼 정리합니다.

```bash
git switch main
git pull --ff-only origin main
git fetch --prune
git branch -d feat/wisdm-evaluation
```

Squash merge 후 `-d`가 거절되면 GitHub의 Merged 표시와 변경 반영, 미푸시 작업이 없는지 확인한 뒤 로컬 브랜치를 정리합니다.

## 검사가 대기 중이어도 머지할 수 있는가?

승인 수 1명을 충족하면 참고용 검사가 대기 중이라는 이유로 머지를 막지 않습니다. 과거 `review-approval` 상태가 남아 있어도 필수 검사에서 제거된 상태라면 차단 조건이 아닙니다. 실제 오류는 확인하고 수정합니다.

실험의 재현 조건과 검증 결과는 승인 규칙과 별개로 PR에 계속 기록합니다. 관련 기준은 [실험 규약](experiment-protocol.md)을 참고합니다.
