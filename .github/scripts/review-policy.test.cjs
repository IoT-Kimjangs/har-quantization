const test = require('node:test');
const assert = require('node:assert/strict');
const check = require('./review-policy.cjs');

async function run({ designated = ['reviewer'], reviews = [], permission = 'write', draft = false } = {}) {
  const failures = [];
  const pr = { number: 1, state: 'open', draft, user: { login: 'author' }, base: { sha: 'base' }, head: { sha: 'head' } };
  const github = { rest: {
    pulls: { get: async () => ({ data: pr }), listReviews: () => {} },
    repos: {
      getContent: async args => {
        assert.equal(args.ref, 'base');
        return { data: { content: Buffer.from(JSON.stringify({ authors: { author: designated } })).toString('base64') } };
      },
      getCollaboratorPermissionLevel: async () => ({ data: { permission } })
    }
  }, paginate: async () => reviews };
  await check({ github, context: { repo: { owner: 'org', repo: 'repo' }, payload: { pull_request: pr } }, core: { setFailed: message => failures.push(message), info: () => {} } });
  return failures;
}
const approval = { id: 1, user: { login: 'reviewer' }, state: 'APPROVED', commit_id: 'head', body: '측정 조건과 결과 파일을 확인했습니다.' };
test('accepts designated reviewer with written approval on current commit', async () => assert.deepEqual(await run({ reviews: [approval] }), []));
test('rejects missing designated reviewers', async () => assert.equal((await run({ designated: [] })).length, 1));
test('rejects self review', async () => assert.equal((await run({ designated: ['author'] })).length, 1));
test('rejects approval on old commit', async () => assert.equal((await run({ reviews: [{ ...approval, commit_id: 'old' }] })).length, 1));
test('rejects blank approval', async () => assert.equal((await run({ reviews: [{ ...approval, body: '  ' }] })).length, 1));
test('rejects review from someone else', async () => assert.equal((await run({ reviews: [{ ...approval, user: { login: 'other' } }] })).length, 1));
test('rejects subsequent change request', async () => assert.equal((await run({ reviews: [approval, { ...approval, id: 2, state: 'CHANGES_REQUESTED' }] })).length, 1));
test('rejects dismissed approval', async () => assert.equal((await run({ reviews: [{ ...approval, state: 'DISMISSED' }] })).length, 1));
test('allows a comment after approval', async () => assert.deepEqual(await run({ reviews: [approval, { ...approval, id: 2, state: 'COMMENTED' }] }), []));
test('requires every designated reviewer', async () => assert.equal((await run({ designated: ['reviewer', 'second'], reviews: [approval] })).length, 1));
test('rejects read-only reviewer', async () => assert.equal((await run({ reviews: [approval], permission: 'read' })).length, 1));
test('rejects draft PR', async () => assert.equal((await run({ draft: true })).length, 1));

const runPolicy = require('./review-policy-runner.cjs');
async function reportStatus(reviews, draft = false) {
  const statuses = [];
  const pr = { number: 2, state: 'open', draft, user: { login: 'improvv' }, base: { sha: 'trusted-base' }, head: { sha: 'current-head' } };
  const github = { rest: {
    pulls: { get: async () => ({ data: pr }), listReviews: () => {} },
    repos: {
      getContent: async args => {
        assert.equal(args.ref, 'trusted-base');
        return { data: { content: Buffer.from(JSON.stringify({ authors: { improvv: ['first', 'second'] } })).toString('base64') } };
      },
      getCollaboratorPermissionLevel: async () => ({ data: { permission: 'write' } }),
      createCommitStatus: async status => statuses.push(status)
    }
  }, paginate: async () => reviews };
  // GitHub Actions exposes repo as an inherited getter, not an own property.
  const actionContext = Object.create({ get repo() { return { owner: 'org', repo: 'repo' }; } });
  actionContext.payload = { pull_request: pr };
  await runPolicy({ github, context: actionContext, core: { info: () => {} } });
  assert.equal(statuses.length, 1);
  assert.equal(statuses[0].sha, 'current-head');
  assert.equal(statuses[0].context, 'review-approval');
  return statuses[0].state;
}
const writtenApproval = (login, id) => ({ id, user: { login }, state: 'APPROVED', commit_id: 'current-head', body: '내용과 실행 조건을 검토했습니다.' });
test('reports pending while two designated reviewers have not approved', async () => assert.equal(await reportStatus([]), 'pending'));
test('reports pending with only one of two approvals', async () => assert.equal(await reportStatus([writtenApproval('first', 1)]), 'pending'));
test('reports success only after both reviewers approve current commit in writing', async () => assert.equal(await reportStatus([writtenApproval('first', 1), writtenApproval('second', 2)]), 'success'));
test('reports pending after new commit invalidates an approval', async () => assert.equal(await reportStatus([writtenApproval('first', 1), { ...writtenApproval('second', 2), commit_id: 'previous-head' }]), 'pending'));
test('reports pending for draft without failing the workflow', async () => assert.equal(await reportStatus([], true), 'pending'));
