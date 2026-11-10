# GitHub 운영 설정

## 머지 조건

**PR을 올리고 작성자 외 팀원 한 명이 Approve하면 머지합니다.** GitHub 기본 승인 기능을 사용하며 별도 승인 스크립트는 사용하지 않습니다.

| 항목 | 설정 |
| --- | --- |
| main 변경 | PR 필수 |
| 필요한 승인 | 작성자 외 1명 |
| 리뷰 본문 | 선택 |
| 새 커밋 시 기존 승인 자동 취소 | 사용하지 않음 |
| 마지막 push에 대한 추가 승인 | 사용하지 않음 |
| 최신 main 반영 필수 | 사용하지 않음 |
| 대화 해결 필수 | 사용하지 않음 |
| CI·별도 승인 검사 통과 필수 | 사용하지 않음 |
| main 강제 push·삭제 | 차단 |
| 머지 방식 | Squash |
| 머지 후 작업 브랜치 | 자동 삭제 |

실제 충돌과 Draft 상태는 GitHub의 기본 절차에 따라 해결합니다. 다른 PR이 먼저 머지됐다는 이유만으로 Update branch와 재승인을 강제하지 않습니다.

## 리뷰어 지정

현재 세 팀원은 `improvv`, `20232541-gaeun`, `20327135`입니다. 작성자 외 리뷰어 한 명을 지정합니다. 서진정(`improvv`)의 PR은 장은성(`20327135`)에게만 요청합니다. `.github/reviewers.json`은 리뷰 후보 안내용이며 후보 모두의 승인이 필수인 것은 아닙니다. 계정별로 별도 승인 상태를 계산하거나 리뷰 본문·승인 커밋을 검사하지 않습니다.

GitHub 기본 규칙은 쓰기 권한이 있는 작성자 외 승인자 1명을 요구합니다. 자기 PR의 승인은 인정되지 않습니다. 리뷰 요청 대상과 필요한 승인 수는 서로 다른 설정입니다.

## 설정 명세와 실제 적용

- ruleset: `main-review-required`, ID `25936545`
- 대상: `refs/heads/main`, Active, 우회 목록 없음
- 승인 설정: `.github/ruleset.json`
- Squash·브랜치 자동 삭제: `.github/repository-settings.json`

이 파일을 커밋하는 것만으로 GitHub 운영 설정이 바뀌지는 않습니다. 관리 권한으로 Settings → Rulesets에서 실제 값을 저장하거나 아래 명령으로 적용합니다.

```bash
bash scripts/apply-github-settings.sh
```

승인 수를 변경할 때는 GitHub 운영 설정을 따로 저장하고 Required approvals 값을 확인합니다. 저장소의 JSON만 바꾸거나 리뷰어 수만 줄여서는 실제 승인 조건이 바뀌지 않습니다.

## 참고용 저장소 검사

`repository-check`는 메타데이터 검사를 실행합니다. 필수 머지 검사로 등록하지 않습니다. 기존 `review-approval` 상태 검사와 `PR review policy` 워크플로는 제거합니다. 과거 검사 기록은 GitHub에 남을 수 있지만 필수 조건에 포함하지 않습니다.
