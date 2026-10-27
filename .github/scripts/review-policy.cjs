module.exports = async ({ github, context, core }) => {
  const pr = context.payload.pull_request;
  if (!pr) return core.setFailed('PR 문맥이 없습니다.');
  const { owner, repo } = context.repo;
  const number = pr.number;
  const current = (await github.rest.pulls.get({ owner, repo, pull_number: number })).data;
  if (current.state !== 'open') return;
  if (current.draft) return core.setFailed('Draft를 해제한 뒤 리뷰를 요청해주세요.');
  // 정책은 PR의 수정본이 아니라 대상 브랜치에서 읽습니다.
  const file = (await github.rest.repos.getContent({
    owner, repo, path: '.github/reviewers.json', ref: current.base.sha
  })).data;
  const config = JSON.parse(Buffer.from(file.content, 'base64').toString('utf8'));
  const author = current.user.login.toLowerCase();
  const mapping = Object.fromEntries(Object.entries(config.authors || {}).map(([k, v]) => [k.toLowerCase(), v]));
  const reviewers = [...new Set((mapping[author] || config.default_reviewers || []).map(x => x.toLowerCase()))];
  if (!reviewers.length) return core.setFailed('지정 리뷰어가 없습니다. 대상 브랜치의 reviewers.json을 먼저 설정해주세요.');
  if (reviewers.includes(author)) return core.setFailed('작성자는 지정 리뷰어가 될 수 없습니다.');
  const reviews = await github.paginate(github.rest.pulls.listReviews, { owner, repo, pull_number: number, per_page: 100 });
  const failures = [];
  for (const login of reviewers) {
    const permission = (await github.rest.repos.getCollaboratorPermissionLevel({ owner, repo, username: login })).data.permission;
    if (!['write', 'maintain', 'admin'].includes(permission)) {
      failures.push(`${login}: 저장소 쓰기 권한이 필요합니다.`);
      continue;
    }
    // 일반 댓글 리뷰는 기존 승인을 대체하지 않습니다. 승인/수정 요청/승인 취소의 최신 상태를 봅니다.
    const latest = reviews.filter(r => r.user.login.toLowerCase() === login && ['APPROVED', 'CHANGES_REQUESTED', 'DISMISSED'].includes(r.state))
      .sort((a, b) => b.id - a.id)[0];
    if (!latest || latest.state !== 'APPROVED' || latest.commit_id !== current.head.sha || !latest.body?.trim()) {
      failures.push(`${login}: 최신 커밋에 검토 내용을 적은 Approve 리뷰가 필요합니다.`);
    }
  }
  if (failures.length) return core.setFailed(failures.join('\n'));
  core.info(`지정 리뷰어 승인 확인: ${reviewers.join(', ')}`);
};
