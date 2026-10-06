import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const read = (path) => readFileSync(join(root, path), 'utf8');
const recipe = (component) => read(`infra/selfhost/Dockerfile.${component}`);
const source = 'https://github.com/he1016060110/GEORank';
const parents = {
  api: 'sha256:0e304874320ff6093df1e71015a99e443793b66259b998c04de317701c7647cd',
  crawler: 'sha256:d8d7b319d7f4999886af8d91984eccf383b2ebc7b9dbdd1ecfabe0764a70a65a',
  frontend: 'sha256:65645c7bb6a0661892a8b03b89d0743208a18dd2f3f17a54ef4b76fb8e2f2a10',
};

for (const component of Object.keys(parents)) {
  test(`${component}: image is annotated with fork URL, full revision and runtime parent`, () => {
    const text = recipe(component);
    assert.ok(text.includes(`ARG SOURCE_REPOSITORY=${source}`));
    assert.match(text, /^ARG SOURCE_REVISION\s*$/m);
    assert.match(text, /^ARG RUNTIME_DEPENDENCIES_SHA256\s*$/m);
    assert.match(text, /test "\$\{#RUNTIME_DEPENDENCIES_SHA256\}" -eq 64/);
    assert.match(text, /case "\$RUNTIME_DEPENDENCIES_SHA256" in \*\[!0-9a-f\]\*\) exit 1/);
    assert.match(text, /org\.yysensor\.georank\.runtime-dependencies\.sha256="\$\{RUNTIME_DEPENDENCIES_SHA256\}"/);
    assert.match(text, /test "\$\{#SOURCE_REVISION\}" -eq 40/);
    assert.match(text, /\*\[!0-9a-f\]\*\) exit 1/);
    assert.match(text, /org\.opencontainers\.image\.source="\$\{SOURCE_REPOSITORY\}"/);
    assert.match(text, /org\.opencontainers\.image\.revision="\$\{SOURCE_REVISION\}"/);
    assert.ok(text.includes(`ARG RUNTIME_IMAGE_ID=${parents[component]}`));
    assert.match(text, /io\.georank\.build\.runtime-image-id="\$\{RUNTIME_IMAGE_ID\}"/);
    assert.match(text, /io\.georank\.build\.source-policy="frozen-git-archive-required"/);
  });

  test(`${component}: no network dependency installers or mutable application mounts`, () => {
    const text = recipe(component);
    assert.doesNotMatch(text, /(?:apt(?:-get)?|pip|npm|pnpm|yarn)\s+(?:install|update)|playwright\s+install|curl\s+https?:|wget\s+/);
    assert.doesNotMatch(text, /^VOLUME\s|--mount=type=bind|COPY\s+\.\s/m);
    assert.doesNotMatch(text, /^#\s*syntax=/m);
  });
}

for (const component of ['api', 'crawler']) {
  test(`${component}: clears inherited code before copying the complete fork backend`, () => {
    const text = recipe(component);
    const cleanup = text.indexOf('root.iterdir()');
    const copy = text.indexOf('COPY backend/ /app/');
    assert.ok(cleanup > 0 && cleanup < copy);
    assert.match(text, /site\.getsitepackages\(\)/);
    assert.match(text, /shutil\.rmtree\(p\)/);
    assert.doesNotMatch(text, /rm -rf\s+\/usr\/|rmtree\([^)]*site-packages/);
    assert.match(text, /COPY --chmod=0755 backend\/docker-entrypoint\.sh \/usr\/local\/bin\/georank-entrypoint/);
    assert.match(text, /io\.georank\.build\.entrypoint-normalization="crlf-to-lf"/);
    assert.ok(text.includes("replace(b'\\r\\n', b'\\n')"));
    assert.match(text, /python -m compileall -q \/app\/app/);
    assert.ok(text.includes("hashlib.sha256(Path('/app/requirements.txt').read_bytes()).hexdigest() == sys.argv[1]"));
    assert.match(text, /ENTRYPOINT \["\/usr\/local\/bin\/georank-entrypoint"\]/);
  });
}

test('api: frozen dependency alias and uvicorn entrypoint', () => {
  assert.match(recipe('api'), /ARG RUNTIME_IMAGE=georank-deps-api:20261006-0e304874320f/);
  assert.match(recipe('api'), /"uvicorn", "app\.main:app"/);
});

test('crawler: cached browser runtime and dedicated crawl queue', () => {
  assert.match(recipe('crawler'), /ARG RUNTIME_IMAGE=georank-deps-crawler:20261006-d8d7b319d7f4/);
  assert.match(recipe('crawler'), /"-Q", "crawl"/);
});

test('frontend: only pinned Nginx runtime plus fork dist and configuration', () => {
  const text = recipe('frontend');
  assert.ok(text.includes(`ARG RUNTIME_IMAGE=nginx:1.27.5-alpine@${parents.frontend}`));
  assert.ok(text.indexOf('rm -rf /usr/share/nginx/html /etc/nginx/conf.d') < text.indexOf('COPY dist/'));
  assert.match(text, /COPY dist\/ \/usr\/share\/nginx\/html\//);
  assert.match(text, /COPY infra\/nginx\/default\.conf \/etc\/nginx\/conf\.d\/default\.conf/);
  assert.doesNotMatch(text, /COPY (?:runtime|data|backend)\//);
  assert.ok(text.includes("sha256sum /etc/nginx/conf.d/default.conf | cut -d ' ' -f 1"));
  assert.ok(text.includes('= "$RUNTIME_DEPENDENCIES_SHA256"'));
});

// This implements only the glob forms intentionally used by .dockerignore.
// It is a bounded contract check, not a replacement for BuildKit context inspection.
function glob(pattern) {
  let regex = '^';
  for (let i = 0; i < pattern.length; i += 1) {
    const ch = pattern[i];
    if (ch === '*' && pattern[i + 1] === '*') {
      i += 1;
      if (pattern[i + 1] === '/') {
        i += 1;
        regex += '(?:.*/)?';
      } else regex += '.*';
    } else if (ch === '*') regex += '[^/]*';
    else if (ch === '?') regex += '[^/]';
    else if (ch === '[') {
      const close = pattern.indexOf(']', i + 1);
      assert.ok(close > i, `Unclosed character class: ${pattern}`);
      regex += pattern.slice(i, close + 1);
      i = close;
    } else regex += ch.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
  return new RegExp(`${regex}$`);
}
const rules = read('.dockerignore').split(/\r?\n/)
  .map((line) => line.trim())
  .filter((line) => line && !line.startsWith('#'))
  .map((line) => {
    const negate = line.startsWith('!');
    return { negate, pattern: glob((negate ? line.slice(1) : line).replace(/^\/+|\/+$/g, '')) };
  });
function allowed(path) {
  let include = true;
  const parts = path.split('/');
  const ancestors = parts.map((_, index) => parts.slice(0, index + 1).join('/'));
  for (const rule of rules) {
    if (ancestors.some((prefix) => rule.pattern.test(prefix))) include = rule.negate;
  }
  return include;
}

test('build context is deny-by-default with no broad COPY of the workspace', () => {
  assert.equal(rules[0].pattern.source, '^.*$');
  assert.equal(rules[0].negate, false);
});

for (const path of [
  'backend/app/tasks/process.py',
  'backend/app/services/company_source.py',
  'backend/app/scripts/migrate.py',
  'backend/requirements.txt',
  'backend/alembic.ini',
  'backend/migration_contracts/example.yaml',
  'backend/docker-entrypoint.sh',
  'dist/admin/settings.html',
  'dist/js/common.js',
  'dist/css/style.css',
  'infra/nginx/default.conf',
  'infra/selfhost/Dockerfile.api',
]) {
  test(`build context includes source: ${path}`, () => assert.equal(allowed(path), true));
}

for (const path of [
  '.git/config',
  '.env',
  '.env.production',
  'docs/customer-notes.md',
  'data/company.json',
  'runtime/private/company.json',
  '.agent/local-secrets/access-token',
  'backend/.env',
  'backend/.env.local',
  'backend/app/secrets/config.json',
  'backend/app/private/company.json',
  'backend/runtime/config.json',
  'backend/data/live.json',
  'backend/auth.json',
  'backend/cookies.txt',
  'backend/cookies-session.json',
  'backend/service-credentials.json',
  'backend/private.key',
  'backend/private.pem',
  'backend/id_rsa',
  'backend/exports/customer.csv',
  'backend/artifacts/screenshot.png',
  'backend/node_modules/library/index.js',
  'backend/.venv/lib/module.py',
  'backend/__pycache__/module.cpython.pyc',
  'backend/test-results/screenshot.png',
  'backend/tmp/company.json',
  'backend/old.sqlite',
  'backend/customer.db',
  'backend/database.dump',
  'backend/celerybeat-schedule',
  'dist/private/customer.json',
  'dist/data/company.json',
  'dist/artifacts/result.html',
]) {
  test(`build context rejects state or credentials: ${path}`, () => assert.equal(allowed(path), false));
}

// These mock assertions check the release contract, not a running container.
function verifyMockRelease(labels, manifest) {
  const digestKey = 'org.yysensor.georank.runtime-dependencies.sha256';
  assert.match(labels[digestKey] || '', /^[0-9a-f]{64}$/);
  assert.equal(labels[digestKey], manifest.runtimeDependenciesSha256);
  assert.equal(labels['org.opencontainers.image.revision'], manifest.sourceRevision);
}
const mockManifest = { sourceRevision: '1'.repeat(40), runtimeDependenciesSha256: 'a'.repeat(64) };
const mockLabels = {
  'org.opencontainers.image.revision': mockManifest.sourceRevision,
  'org.yysensor.georank.runtime-dependencies.sha256': mockManifest.runtimeDependenciesSha256,
};
test('release mock: dependency source-contract label exactly matches manifest', () => {
  assert.doesNotThrow(() => verifyMockRelease(mockLabels, mockManifest));
});
for (const invalid of ['', 'a'.repeat(63), 'A'.repeat(64), 'g'.repeat(64), 'b'.repeat(64)]) {
  test('release mock: missing, malformed or mismatched digest fails closed: ' + (invalid || 'empty'), () => {
    assert.throws(() => verifyMockRelease({ ...mockLabels, 'org.yysensor.georank.runtime-dependencies.sha256': invalid }, mockManifest));
  });
}
