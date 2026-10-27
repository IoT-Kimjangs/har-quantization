# GitHub 운영 설정

이 저장소의 `main`은 검토를 마친 변경만 받습니다. 문서·분석·실험 코드 모두 같은 PR 규칙을 적용합니다.

## 적용 상태

2026-10-27 저장소를 생성하고 `main-review-required` ruleset(ID 25936545)을 Active로 저장했습니다. `repository-check`와 `review-approval`는 GitHub Actions 출처로 제한했습니다. 최소 승인 2명, 새 커밋 시 재승인, 마지막 푸시 외 승인, 대화 해결, main 삭제·강제 푸시 차단을 적용했습니다. 저장소 검사는 초기 커밋에서 통과했습니다. 2026-10-27 팀원 세 계정을 확인해 작성자별 다른 두 명을 리뷰어로 등록했고, 두 팀원에게 저장소 Write 권한을 부여했습니다. 승인 대기는 Pending으로 표시합니다. 실제 승인 후 머지 검증은 아직 수행하지 않았습니다.

## 저장소 설정

- Squash merge만 허용
- 머지 후 원격 작업 브랜치 자동 삭제
- 기본 브랜치 변경 시 작업 브랜치 업데이트 제안
- Squash 커밋은 PR 제목과 본문을 기준으로 실제 변경 내용을 정리

API 설정값은 `.github/repository-settings.json`에 기록합니다. 이 파일을 커밋하는 것만으로 GitHub 설정이 바뀌지는 않습니다.

## main ruleset

- 대상: `refs/heads/main`, 활성화
- 관리자·앱을 포함해 우회 허용 목록 없음
- 직접 푸시 대신 PR 필수, 작성자 외 최소 2명 승인
- 추가 커밋이 생기면 이전 승인 무효
- 마지막 푸시한 사람과 다른 사람의 승인 필수
- 모든 리뷰 대화 해결 필수
- `repository-check`, `review-approval` 통과 필수
- 최신 main과 동기화 후 검사
- main 삭제·강제 푸시 금지, 선형 커밋 이력 유지

명세는 `.github/ruleset.json`입니다. 새 저장소의 최초 파일 등록은 main 생성에 필요합니다. 최초 등록 후부터 위 보호 규칙을 적용하며, 이후 설정 변경도 PR로 검토합니다.

## 지정 리뷰어

`.github/reviewers.json`에 작성자별 GitHub 아이디를 소문자로 등록합니다. `authors`에 작성자가 있으면 해당 배열의 리뷰어 전원, 없으면 `default_reviewers` 전원이 승인해야 합니다. 작성자 자신은 배열에 넣지 않습니다. 빈 배열은 승인을 면제하지 않고 승인 검사를 Pending 상태로 유지합니다.

설정은 PR 대상 브랜치의 파일을 읽습니다. 리뷰어는 쓰기 이상의 권한이 있어야 하며 최신 PR 커밋에 검토 내용을 적은 Approve 리뷰를 제출해야 합니다. 새 커밋, 승인 취소, 수정 요청, 비어 있는 리뷰 내용은 통과하지 않습니다. 일반 댓글은 Approve를 대신하지 않습니다.

GitHub 기본 승인 규칙은 승인 수와 코드 소유자 검토를 확인하지만 리뷰 본문을 필수로 만들지는 않습니다. 그래서 별도 `review-approval` 검사를 함께 사용합니다. 검사 재실행은 새 커밋 또는 리뷰 제출·수정·취소로 이루어집니다. 정책 워크플로는 PR 코드를 실행하지 않고 대상 브랜치의 검사 코드만 읽습니다.

리뷰어 작성자별 리뷰어를 등록했습니다. 작성자를 제외한 나머지 팀원 두 명의 승인을 요구합니다. GitHub의 Reviewers 선택도 함께 해주세요.

## 적용 및 확인

관리 권한이 있는 계정으로 다음 명령을 실행하면 설정 명세를 적용할 수 있습니다. ruleset은 같은 이름이 있으면 갱신합니다.

```bash
bash scripts/apply-github-settings.sh
```

Settings → General에서 squash와 자동 삭제를, Settings → Rulesets에서 활성 ruleset·main 패턴·우회 목록을 확인합니다. PR을 만들고 리뷰 전 머지 차단, 지정 리뷰어의 내용 있는 승인 후 검사 성공, 새 커밋 후 재차 차단을 확인합니다.

이 설정은 GitHub API 또는 UI에 저장돼야 강제됩니다. JSON 파일과 협업 문서만으로는 보호되지 않습니다. GitHub 관리자 권한으로 규칙 자체를 수정할 수 있는 점과 일반 머지 우회는 구분합니다.

참고: [GitHub ruleset 규칙](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)

## 승인 대기와 실행 실패 구분

`review-policy-update` 워크플로는 승인 요건을 확인해 PR 최신 커밋의 `review-approval` 상태를 갱신합니다. 미승인·Draft·재리뷰 대기는 Pending이며 워크플로 실패 메일을 발생시키는 실행 오류로 처리하지 않습니다. 두 리뷰어 전원이 승인하면 Success입니다. 정책 파일은 대상 브랜치에서 읽고 PR의 코드나 스크립트를 실행하지 않습니다. GitHub 토큰에는 상태 갱신에 필요한 statuses:write만 추가했습니다.

초기 정책의 빈 리뷰어 목록과 승인 대기를 실패로 처리하던 동작은 관리 설정 초기화로 수정했습니다. PR #2는 머지하지 않고 두 리뷰어의 검토를 기다립니다.
