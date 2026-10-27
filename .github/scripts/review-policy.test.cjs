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
