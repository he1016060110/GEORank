// Fork modification (he1016060110, 2026-10-06): generic offline admin UTC fixture.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { spawnSync } from 'node:child_process';

// Extract actual production helper functions; never import credentials, perform
// localhost requests, write company records or dispatch real/model work.
const source = await readFile(new URL('../dist/js/admin.js', import.meta.url), 'utf8');
const dateHelpers = source.slice(source.indexOf('    function parseAdminDate(value) {'), source.indexOf('    function escapeHtml(value) {'));
const now = Date.parse('2026-10-06T02:11:31.000Z');
const api = vm.runInNewContext(`${dateHelpers}\n({parseAdminDate, formatDate, timeAgo});`, {
  Date: class FixtureDate extends Date { static now() { return now; } },
}, { timeout: 1000 });

test('same-origin constant and fork attribution remain; whole production script compiles', () => {
  assert.match(source, /^\/\/ Fork modification /);
  assert.match(source, /const API_BASE = '';/);
  assert.doesNotThrow(() => new vm.Script(source));
});

test('UTC-naive full timestamp has the same instant and relative display as Z timestamp', () => {
  const naive = '2026-10-06T02:11:30';
  const aware = `${naive}Z`;
  assert.equal(api.parseAdminDate(naive).getTime(), Date.parse(aware));
  assert.equal(api.timeAgo(naive), '刚刚');
  assert.equal(api.timeAgo(naive), api.timeAgo(aware));
  assert.equal(api.formatDate(naive), api.formatDate(aware));
  assert.equal(api.timeAgo('2026-10-06T02:06:31'), '5 分钟前');
});

test('naive space separator and fractional seconds are UTC, explicit offsets remain intact', () => {
  for (const naive of ['2026-10-06 02:11:30', '2026-10-06T02:11:30.123', '2026-10-06 02:11:30.123456']) {
    const equivalent = `${naive.replace(' ', 'T')}Z`;
    assert.equal(api.parseAdminDate(naive).getTime(), Date.parse(equivalent));
    assert.equal(api.timeAgo(naive), api.timeAgo(equivalent));
  }
  for (const aware of ['2026-10-06T10:11:30+08:00', '2026-10-06T10:11:30+0800', '2026-10-06T02:11:30Z', '2026-10-06T01:11:30-01:00']) {
    assert.equal(api.parseAdminDate(aware).getTime(), Date.parse(aware));
    assert.equal(api.timeAgo(aware), '刚刚');
  }
});

test('date-only retains pre-existing Date display behavior; invalid values render -- not NaN', () => {
  const dateOnly = '2026-10-06';
  const original = new Date(dateOnly);
  assert.equal(api.parseAdminDate(dateOnly).getTime(), original.getTime());
  assert.equal(api.formatDate(dateOnly), original.toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' }));
  for (const invalid of [null, undefined, '', ' ', 'not-a-date', '2026-13-06T02:11:30', '2026-10-06T25:11:30', '2026-10-06T02:99:30', '2026-10-06T02:11:30+99:00']) {
    assert.equal(api.parseAdminDate(invalid), null);
    assert.equal(api.formatDate(invalid), '--');
    assert.equal(api.timeAgo(invalid), '--');
  }
});

test('timezone regression reproduced and fixed under Asia/Shanghai and negative offset zones', () => {
  for (const timezone of ['Asia/Shanghai', 'America/Los_Angeles', 'UTC']) {
    const script = `${dateHelpers}\nconst now = Date.parse('2026-10-06T02:11:31Z'); Date.now = () => now;\nif (timeAgo('2026-10-06T02:11:30') !== '刚刚') throw Error('naive relative time shifted');\nif (parseAdminDate('2026-10-06T10:11:30+08:00').getTime() !== Date.parse('2026-10-06T02:11:30Z')) throw Error('offset changed');\nif (formatDate('2026-10-06') !== new Date('2026-10-06').toLocaleDateString('zh-CN', {year:'numeric',month:'2-digit',day:'2-digit'})) throw Error('date-only changed');`;
    const result = spawnSync(process.execPath, ['--input-type=module', '-e', script], { env: { ...process.env, TZ: timezone }, encoding: 'utf8', timeout: 5000 });
    assert.equal(result.status, 0, `${timezone}: ${result.stderr}`);
  }
});
