# 협업 규칙

모든 팀원은 작업 브랜치에서 커밋하고 `main`을 대상으로 PR을 올립니다. 작성자 외 지정 리뷰어가 내용을 검토하고 승인한 뒤 머지합니다.

화면별 절차와 본문 예시는 [PR 작성 안내](docs/pr-guide.md), 주간 작업과 마감은 [주차별 수행 계획](docs/project-plan.md)을 확인합니다.

## 작업 순서

```bash
git switch main
git pull --ff-only origin main
git switch -c feat/latency-benchmark
# 파일 수정 및 확인
git add <수정한-파일>
git commit -m "feat: TFLite 지연시간 측정 추가"
git push -u origin feat/latency-benchmark
```

PR은 변경 목적, 실제 바뀐 내용, 확인 방법, 결과를 적고 리뷰어를 지정합니다. 작업 중이면 Draft로 올립니다. 하나의 PR에는 함께 검토할 수 있는 작업만 담습니다.

## 브랜치와 커밋

- 기본 브랜치: `main`
- 작업 브랜치: `feat/`, `fix/`, `docs/`, `exp/`, `chore/` 다음에 짧은 영문 작업명
- 커밋: `feat:`, `fix:`, `docs:`, `exp:`, `chore:` 다음에 구체적인 변경 내용
- 예: `docs: WISDM 데이터 준비 방법 정리`, `fix: 워밍업 시간을 측정값에서 제외`
- 의미 없는 일괄 추가를 피하고 `git diff --cached`로 포함될 파일을 확인합니다.
- 공용 브랜치의 이력을 덮어쓰지 않습니다. 개인 브랜치 이력 정리가 필요하면 협의 후 `--force-with-lease`를 사용합니다.

## 리뷰와 승인

1. `.github/reviewers.json`에 등록된 작성자별 지정 리뷰어를 PR의 Reviewers에 추가합니다.
2. 지정 리뷰어 전원이 최신 커밋을 확인하고, 검토 내용을 적은 **Approve 리뷰**를 제출합니다. 일반 댓글이나 `LGTM` 댓글만으로는 승인되지 않습니다.
3. 작성자는 자신의 PR을 승인할 수 없습니다. 마지막 변경을 푸시한 사람과 다른 사람이 승인해야 합니다.
4. 코드가 추가되면 이전 승인은 무효이며 다시 리뷰받습니다.
5. 수정 요청을 반영하고 모든 리뷰 대화를 해결합니다. 작성자가 요청을 임의로 무시하거나 승인을 대신하지 않습니다.
6. 필수 검사 `review-approval`과 `repository-check`가 성공해야 합니다. 지정 리뷰어 승인 전 `review-approval`은 Pending으로 표시됩니다.
7. 조건이 충족되면 **Squash and merge**합니다. 제목은 최종 변경을 설명하고, 본문에는 실제 변경·검증 내용만 남깁니다.
8. 머지된 원격 작업 브랜치는 자동 삭제합니다. 로컬은 아래처럼 정리합니다.

```bash
git switch main
git pull --ff-only origin main
git fetch --prune
git branch -d feat/latency-benchmark
```

지정 리뷰어 변경은 사유를 적은 별도 PR로 처리합니다. 리뷰어가 없거나 작성자 자신이 포함되면 승인 검사가 Pending 상태로 남아 머지를 막습니다. GitHub 설정 적용 전에는 문서만으로 머지를 차단할 수 없으므로 [운영 설정](docs/github-rules.md)의 적용 상태를 확인합니다.

## 실험 PR

데이터 출처·분할·seed, 모델 출처와 해시, OS·CPU·메모리, 패키지 버전, 스레드·delegate, 실행 명령, 원시 측정 기록을 함께 제출합니다. 무작위 입력 지연시간은 정확도 평가와 구분합니다. 숫자가 기대와 달라도 제거하지 말고 원인을 기록합니다.

원본 데이터, 대용량 모델, 비밀번호, 토큰, 개인 절대경로는 커밋하지 않습니다. 공개할 수 있는 요약 결과와 재현 방법을 남깁니다.
