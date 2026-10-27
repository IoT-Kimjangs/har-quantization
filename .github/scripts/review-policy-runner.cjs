const check = require('./review-policy.cjs');

module.exports = async ({ github, context, core }) => {
  const { owner, repo } = context.repo;
  const prs = context.payload.pull_request
    ? [(await github.rest.pulls.get({ owner, repo, pull_number: context.payload.pull_request.number })).data]
    : await github.paginate(github.rest.pulls.list, { owner, repo, state: 'open', base: 'main', per_page: 100 });
  for (const pr of prs) {
    if (pr.state !== 'open') continue;
    const pending = [];
    await check({ github, context: { ...context, payload: { ...context.payload, pull_request: pr } }, core: {
      info: message => core.info(message),
      setFailed: message => { pending.push(message); core.info(message); }
    } });
    await github.rest.repos.createCommitStatus({
      owner, repo, sha: pr.head.sha, context: 'review-approval',
      state: pending.length ? 'pending' : 'success',
      description: pending.length ? '지정 리뷰어의 최신 커밋 승인 대기' : '지정 리뷰어 전원 승인 확인',
      target_url: `https://github.com/${owner}/${repo}/pull/${pr.number}`
    });
  }
};
