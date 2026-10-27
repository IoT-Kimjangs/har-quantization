#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
repo="IoT-Kimjangs/har-quantization"
gh api --method PATCH "repos/$repo" --input .github/repository-settings.json
ruleset_id="$(gh api "repos/$repo/rulesets" --jq '.[] | select(.name == "main-review-required") | .id')"
if [[ -n "$ruleset_id" ]]; then
  gh api --method PUT "repos/$repo/rulesets/$ruleset_id" --input .github/ruleset.json
else
  gh api --method POST "repos/$repo/rulesets" --input .github/ruleset.json
fi
gh api "repos/$repo" --jq '{default_branch,allow_merge_commit,allow_squash_merge,allow_rebase_merge,delete_branch_on_merge}'
gh api "repos/$repo/rulesets" --jq '.[] | {name,enforcement,id}'
