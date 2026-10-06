# 从自有 Fork 冻结源码并构建自托管镜像

本说明只定义通用的构建和发布合同；不包含客户资料、真实密钥、数据库快照或审核结果，也不是已完成部署的证明。

## 1. 构建范围与证据边界

- 应用真源：`https://github.com/he1016060110/GEORank` 的一个完整 40 位 Git commit。
- 构建上下文必须由该 commit 的 `git archive` 导出，不能直接把含未提交补丁、数据库和 `.env` 的工作目录作为发布上下文。
- `infra/selfhost/Dockerfile.api` 和 `Dockerfile.crawler` 复用经过核验的缓存依赖镜像，只保留 `/app` 外的 Python 包、系统工具及 Playwright 浏览器；构建时先清空整个旧 `/app`，再复制 Fork 的完整 `backend/` 闭包。
- Fork 的 `backend/docker-entrypoint.sh` 同时覆盖 `/app/docker-entrypoint.sh` 和 `/usr/local/bin/georank-entrypoint`。唯一声明的入口转换是 CRLF → LF，以支持 Windows checkout。
- `Dockerfile.frontend` 只采用 digest 固定的 Nginx 运行时，先清除旧 HTML/config，再复制同一 commit 的 `dist/` 和 `infra/nginx/default.conf`。
- 所有镜像写入 `org.opencontainers.image.source` / `org.opencontainers.image.revision` 和依赖父镜像的完整 ID；Dockerfile 拒绝空或不是 40 位小写十六进制的 revision。
- 所有镜像还必须接收 `RUNTIME_DEPENDENCIES_SHA256`，要求 64 位小写十六进制，写为 `org.yysensor.georank.runtime-dependencies.sha256`，且必须与发布 manifest 完全一致。API/Crawler 的值为冻结归档 `backend/requirements.txt` 原始字节 SHA-256；Frontend 的值为冻结归档 `infra/nginx/default.conf` 原始字节 SHA-256。Dockerfile 在 COPY 后核对实际文件，不接受仅贴标签而来源文件不同。
- 该字段是 **Fork 源码依赖合同摘要**，不是“重新安装过依赖”或“所有已安装包一一等于 requirements”的声明。缓存父镜像 ID、其实际包清单和源码依赖合同必须分别核验；父文件/checkout 的 CRLF 和 Git archive 的 LF 会产生不同的原始字节哈希，应比较归一换行后的要求是否相同，再使用归档原始字节哈希作为发布标签。
- **这不是“从干净基础镜像重新安装所有依赖”。** 这是“固定依赖运行时 + 完整替换应用源码”。`requirements.txt` 或 Python/Playwright 依赖发生变化时，须先生成、测试并固定新的依赖镜像，不能声称只换源码便完成依赖升级。
- Labels 是审计字段，不会自行证明上下文属于该 commit；冻结归档、父镜像 ID 校验以及最终容器代码哈希必须分别核验。

### 当前可复用依赖运行时

| 组件 | 本机缓存别名 | 必须匹配的父镜像 ID |
| --- | --- | --- |
| API / 默认 Worker / Beat / 迁移 | `georank-deps-api:20261006-0e304874320f` | `sha256:0e304874320ff6093df1e71015a99e443793b66259b998c04de317701c7647cd` |
| 爬虫 Worker | `georank-deps-crawler:20261006-d8d7b319d7f4` | `sha256:d8d7b319d7f4999886af8d91984eccf383b2ebc7b9dbdd1ecfabe0764a70a65a` |
| Frontend | `nginx:1.27.5-alpine@sha256:65645c7bb6a0661892a8b03b89d0743208a18dd2f3f17a54ef4b76fb8e2f2a10` | `sha256:65645c7bb6a0661892a8b03b89d0743208a18dd2f3f17a54ef4b76fb8e2f2a10` |

上述 API/crawler 别名只是便于 BuildKit 读取的本机标签，**不是不可变 ID**。必须在构建前用 `docker image inspect` 比较实际 ID。不要把裸 `sha256:...` 写为 `FROM` 仓库名。

## 2. 源码合同测试

在仓库根目录执行：

```powershell
node --test infra/selfhost/contract.mjs
```

检查包括完整源码复制、旧树清理、入口归属与换行规则、源码/依赖标签、专用爬虫队列、禁止安装时联网、构建上下文 allowlist，以及运行数据/密钥路径 denylist。

这是一组静态定向合同测试。它不替代实际镜像构建、Compose 解析、容器 import/队列验收或真实浏览器验收。

## 3. PowerShell 冻结归档与构建示例

以下命令只构建镜像，不会重启服务、迁移现有数据、运行真实 AI、审核公司或推送上游。发布人员须自行选择准备发布的、已提交的 Fork commit，并先完成相关应用测试。

```powershell
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path '.').Path  # 运行位置必须是自有 Fork 的根目录
$revision = (git -C $repo rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $revision -notmatch '^[0-9a-f]{40}$') {
    throw '无法冻结完整 Git revision'
}
$sourceUrl = 'https://github.com/he1016060110/GEORank'
$sourcePaths = @('backend', 'dist', 'infra/nginx', 'infra/selfhost', '.dockerignore')
git -C $repo diff --quiet -- $sourcePaths
if ($LASTEXITCODE -ne 0) { throw '构建闭包存在未提交修改，请先提交并复核' }
git -C $repo diff --cached --quiet -- $sourcePaths
if ($LASTEXITCODE -ne 0) { throw '构建闭包存在尚未提交的暂存修改' }
$untracked = @(git -C $repo ls-files --others --exclude-standard -- $sourcePaths)
if ($LASTEXITCODE -ne 0 -or $untracked.Count -gt 0) {
    throw '构建闭包存在未跟踪文件，请先明确是否纳入该 commit'
}
node --test (Join-Path $repo 'infra/selfhost/contract.mjs')
if ($LASTEXITCODE -ne 0) { throw 'Fork 构建合同测试失败' }

$apiParent = 'sha256:0e304874320ff6093df1e71015a99e443793b66259b998c04de317701c7647cd'
$crawlerParent = 'sha256:d8d7b319d7f4999886af8d91984eccf383b2ebc7b9dbdd1ecfabe0764a70a65a'
$frontendParent = 'sha256:65645c7bb6a0661892a8b03b89d0743208a18dd2f3f17a54ef4b76fb8e2f2a10'
$apiRuntime = 'georank-deps-api:20261006-0e304874320f'
$crawlerRuntime = 'georank-deps-crawler:20261006-d8d7b319d7f4'

# 创建依赖别名只影响镜像标签，不改变现有容器。
docker image inspect $apiParent --format '{{.Id}}' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'API 依赖镜像未缓存，须先准备依赖运行时' }
docker image inspect $crawlerParent --format '{{.Id}}' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Crawler 依赖镜像未缓存' }
docker image inspect $frontendParent --format '{{.Id}}' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Nginx 运行时未缓存' }
docker tag $apiParent $apiRuntime
if ($LASTEXITCODE -ne 0) { throw '创建 API 依赖别名失败' }
docker tag $crawlerParent $crawlerRuntime
if ($LASTEXITCODE -ne 0) { throw '创建 Crawler 依赖别名失败' }
docker tag $frontendParent 'nginx:1.27.5-alpine'
if ($LASTEXITCODE -ne 0) { throw '创建 Nginx 版本别名失败' }
if ((docker image inspect $apiRuntime --format '{{.Id}}').Trim() -ne $apiParent) {
    throw 'API 依赖别名漂移，停止构建'
}
if ((docker image inspect $crawlerRuntime --format '{{.Id}}').Trim() -ne $crawlerParent) {
    throw 'Crawler 依赖别名漂移，停止构建'
}

# 每次使用全新的临时路径；不执行通配或递归删除。
$work = Join-Path ([IO.Path]::GetTempPath()) ('georank-build-' + [Guid]::NewGuid().ToString('N'))
$context = Join-Path $work 'source'
New-Item -ItemType Directory -Path $context | Out-Null
$archive = Join-Path $work ('georank-' + $revision + '.tar')
git -C $repo archive --format=tar --output=$archive $revision
if ($LASTEXITCODE -ne 0) { throw '源码归档失败' }
tar -xf $archive -C $context
if ($LASTEXITCODE -ne 0) { throw '源码归档展开失败' }
if (-not (Test-Path -LiteralPath (Join-Path $context '.dockerignore'))) {
    throw '冻结归档缺少 .dockerignore，拒绝构建'
}
Get-FileHash -LiteralPath $archive -Algorithm SHA256

$requirementsSha = (Get-FileHash -LiteralPath (Join-Path $context 'backend/requirements.txt') -Algorithm SHA256).Hash.ToLowerInvariant()
$frontendContractSha = (Get-FileHash -LiteralPath (Join-Path $context 'infra/nginx/default.conf') -Algorithm SHA256).Hash.ToLowerInvariant()
# 发布 manifest 的对应两项必须原样使用这两个值。
$short = $revision.Substring(0, 12)
$apiImage = "georank-fork-api:$short"
$crawlerImage = "georank-fork-crawler:$short"
$frontendImage = "georank-fork-frontend:$short"
$commonArgs = @('--pull=false', '--network=none',
    '--build-arg', "SOURCE_REPOSITORY=$sourceUrl",
    '--build-arg', "SOURCE_REVISION=$revision")

docker build @commonArgs --build-arg "RUNTIME_IMAGE=$apiRuntime" `
    --build-arg "RUNTIME_IMAGE_ID=$apiParent" `
    --build-arg "RUNTIME_DEPENDENCIES_SHA256=$requirementsSha" `
    -f (Join-Path $context 'infra/selfhost/Dockerfile.api') -t $apiImage $context
if ($LASTEXITCODE -ne 0) { throw 'Fork API 构建失败' }
docker build @commonArgs --build-arg "RUNTIME_IMAGE=$crawlerRuntime" `
    --build-arg "RUNTIME_IMAGE_ID=$crawlerParent" `
    --build-arg "RUNTIME_DEPENDENCIES_SHA256=$requirementsSha" `
    -f (Join-Path $context 'infra/selfhost/Dockerfile.crawler') -t $crawlerImage $context
if ($LASTEXITCODE -ne 0) { throw 'Fork Crawler 构建失败' }
docker build @commonArgs --build-arg "RUNTIME_IMAGE=nginx:1.27.5-alpine@$frontendParent" `
    --build-arg "RUNTIME_IMAGE_ID=$frontendParent" `
    --build-arg "RUNTIME_DEPENDENCIES_SHA256=$frontendContractSha" `
    -f (Join-Path $context 'infra/selfhost/Dockerfile.frontend') -t $frontendImage $context
if ($LASTEXITCODE -ne 0) { throw 'Fork Frontend 构建失败' }

foreach ($image in @($apiImage, $crawlerImage, $frontendImage)) {
    $source = (docker image inspect $image --format '{{index .Config.Labels "org.opencontainers.image.source"}}').Trim()
    $builtRevision = (docker image inspect $image --format '{{index .Config.Labels "org.opencontainers.image.revision"}}').Trim()
    if ($source -ne $sourceUrl -or $builtRevision -ne $revision) {
        throw "镜像来源标签不一致：$image"
    }
    $builtDependencySha = (docker image inspect $image --format '{{index .Config.Labels "org.yysensor.georank.runtime-dependencies.sha256"}}').Trim()
    $expectedDependencySha = if ($image -eq $frontendImage) { $frontendContractSha } else { $requirementsSha }
    if ($builtDependencySha -ne $expectedDependencySha) { throw "依赖源码合同标签不一致：$image" }
    docker image inspect $image --format 'ID={{.Id}} SOURCE={{index .Config.Labels "org.opencontainers.image.source"}} REVISION={{index .Config.Labels "org.opencontainers.image.revision"}} PARENT={{index .Config.Labels "io.georank.build.runtime-image-id"}}'
}
```

`--network=none` 禁止构建步骤联网；`--pull=false` 不等于“所有 Docker daemon/BuildKit 都绝不查询镜像仓库”。若当前 builder 无法使用已核验本机缓存，须先修复或明确准备该 builder 的运行时，不得悄悄换成 `latest`。Dockerfile 没有远程 `# syntax=` 依赖，避免为解析前端额外下载镜像。

## 4. 部署必须另做，不能只凭 build 成功

1. 冻结现有容器 ID、镜像 ID、Compose 文件顺序、网络、实际 mounts 和只读健康状态。多人并行环境中写入前再次比对；状态已变化就重新合并，不能覆盖他人发布。
2. 在正式发布覆盖层将 API、默认 Worker、Beat、迁移服务指向同一 Fork API 镜像，Crawler 指向 Fork Crawler 镜像，Frontend 指向 Fork Frontend 镜像。
3. 删除这些服务的旧应用源码 bind：`/app`、`/usr/share/nginx/html` 和覆盖源码 Nginx 配置的挂载不得继续遮盖镜像里的 Fork 文件。**保留**环境配置、非源码运行配置、数据库/队列/图谱/向量/对象存储持久卷和当前网络；不能清库来规避版本问题。
4. 只读解析最终 Compose 结果，逐项核对 image 和 mounts，不将含秘密的完整环境展开结果写到公开文档或仓库。发布日志只列非秘密结构和版本信息。
5. 重建需要变更的应用服务，按现有迁移策略处理。不能把其他服务一起切回旧镜像或让新的发布覆盖层被旧 pin 覆盖。
6. 核对运行容器 image ID 与 Fork labels；对 `app/tasks/process.py`、相关 service、入口（按 LF 归一）和静态脚本/Nginx 配置做宿主归档/容器/HTTP 哈希比对。
7. 验证 API import、Celery 注册任务与实际队列、数据库连接和版本、浏览器同源 API 路径与页面时序。仅 health 成功不能证明爬虫/模型/图谱/向量成果或实际前端已更新。
8. 先执行不调用真实 AI 的隔离测试。真实任务需要独立授权，并按保留原 ID、读取状态避免重复任务、验证持久成果、不代替审核发布的合同执行。

### 构建上下文安全边界

根 `.dockerignore` 使用默认拒绝，仅允许 `backend/`、`dist/`、Nginx 配置和构建配方，再拒绝其内部的 `.env`、credentials/cookies、私钥、运行目录、数据库、私有内容、依赖目录、证据与临时文件。`git archive` 不包含 `.git` 或未跟踪文件。两者都必要，但都不能代替提交前的真实密钥/客户正文扫描。

不要把新的客户正文写到 `docs/` 或构建工具目录，不要把真实密钥作为 build-arg，也不要把发布现场的环境文件复制进镜像。
