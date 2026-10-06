// Fork modification (he1016060110, 2026-10-06): same-origin API and verified company pipeline.
/**
 * 提交公司完整分析页
 * - 新标签页展示完整 AI 分析过程
 * - 分析完成后由用户确认提交审核
 */
(function () {
    'use strict';

    const API_BASE = '';

    const stepOrder = ['crawl', 'clean', 'graph', 'vector'];
    const statusMap = {
        pending: { active: 0, completed: -1, note: '已创建任务，等待进入官网解析队列。', activity: '正在创建分析任务并加入处理队列...' },
        crawling: { active: 0, completed: -1, note: '正在抓取官网首页和一级目录，AI 将锁定最值得深入的页面。', activity: '正在解析官网首页并识别一级目录链接...' },
        cleaning: { active: 1, completed: 0, note: '关键页面已锁定，正在抽取企业介绍、产品与团队信息。', activity: '正在整合关键页面内容并结构化抽取企业信息...' },
        graph_building: { active: 2, completed: 1, note: '正在构建知识图谱，各阶段成果以保存的核验回执为准。', activity: '正在构建实体关系与企业知识图谱...' },
        vectorizing: { active: 3, completed: 2, note: '正在写入向量知识库，图谱与向量成果须分别核验。', activity: '正在将企业知识写入语义检索索引...' },
        completed: { active: -1, completed: 3, note: 'AI 分析完成，结果当前仍为草稿，点击提交审核后才会进入后台审核。', activity: '企业知识库构建完成，等待你确认提交审核。' },
        failed: { active: -1, completed: -1, note: '本次分析未通过质量核验，请在后台查看失败阶段并重新分析。', activity: '分析未成功完成。' },
        dispatch_unknown: { active: -1, completed: -1, note: '派发回执尚未确认，任务可能已进入队列；只观察同一任务，不要重复提交。', activity: '等待同一任务的派发与处理回执，当前不代表分析失败。' },
        unknown: { active: -1, completed: -1, note: '记录曾标为完成，但缺少完整质量核验回执；不能认定知识库已完成或提交审核。', activity: '企业资料、图谱和向量成果待核验。' },
    };

    const analysisUrl = document.getElementById('analysis-url');
    const analysisBadge = document.getElementById('analysis-badge');
    const analysisNote = document.getElementById('analysis-note');
    const analysisActivity = document.getElementById('analysis-activity');
    const analysisFeed = document.getElementById('analysis-feed');
    const selectedPagesPanel = document.getElementById('selected-pages-panel');
    const selectedPagesList = document.getElementById('selected-pages-list');
    const selectedPagesCount = document.getElementById('selected-pages-count');
    const companyPreviewCard = document.getElementById('company-preview-card');
    const previewCompanyName = document.getElementById('preview-company-name');
    const previewCompanySummary = document.getElementById('preview-company-summary');
    const reviewStatusTitle = document.getElementById('review-status-title');
    const reviewStatusCopy = document.getElementById('review-status-copy');
    const submitReviewBtn = document.getElementById('submit-review-btn');
    const openAdminReviewBtn = document.getElementById('open-admin-review-btn');
    const refreshAnalysisBtn = document.getElementById('refresh-analysis-btn');

    let pollTimer = null;
    let seenFeedKeys = new Set();
    let currentCompanyId = '';
    let currentNormalizedUrl = '';
    let currentEffectiveStatus = 'pending';
    let currentPublishStatus = 'draft';
    const loadedPreviewIds = new Set();

    function getAuthToken() {
        return localStorage.getItem('georank_user_token')
            || localStorage.getItem('georank_token')
            || '';
    }

    async function request(path, options = {}) {
        const headers = {
            'Content-Type': 'application/json',
            ...(options.headers || {}),
        };
        const token = getAuthToken();
        if (token) {
            headers.Authorization = `Bearer ${token}`;
        }

        const response = await fetch(`${API_BASE}${path}`, {
            ...options,
            headers,
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            const message = data.detail || `请求失败 (${response.status})`;
            throw new Error(typeof message === 'string' ? message : JSON.stringify(message));
        }
        return data;
    }

    function normalizeUrlInput(raw) {
        const value = (raw || '').trim();
        if (!value) return '';
        const withProtocol = value.includes('://')
            ? value
            : value.startsWith('//')
                ? `https:${value}`
                : `https://${value}`;
        try {
            const parsed = new URL(withProtocol);
            if (!['http:', 'https:'].includes(parsed.protocol) || !parsed.hostname) {
                return '';
            }
            parsed.hash = '';
            parsed.search = '';
            if (parsed.pathname === '/') {
                parsed.pathname = '';
            }
            return parsed.toString().replace(/\/$/, '');
        } catch (_) {
            return '';
        }
    }

    function clearPolling() {
        if (pollTimer) {
            clearTimeout(pollTimer);
            pollTimer = null;
        }
    }

    function setStepState(stepName, state) {
        const stepEl = document.querySelector(`#analysis-steps [data-step="${stepName}"]`);
        if (!stepEl) return;
        const iconBg = stepEl.querySelector('.w-10');
        const iconEl = iconBg?.querySelector('.material-symbols-outlined');
        const statusIcon = stepEl.querySelector('.step-icon');

        if (!iconBg || !iconEl || !statusIcon) return;

        if (state === 'done') {
            iconBg.className = 'w-10 h-10 rounded-2xl bg-green-50 flex items-center justify-center shrink-0';
            iconEl.className = 'material-symbols-outlined text-green-600 text-lg';
            statusIcon.textContent = 'check_circle';
            statusIcon.className = 'material-symbols-outlined text-green-600 text-lg step-icon';
            return;
        }
        if (state === 'active') {
            iconBg.className = 'w-10 h-10 rounded-2xl bg-primary/10 flex items-center justify-center shrink-0';
            iconEl.className = 'material-symbols-outlined text-primary text-lg';
            statusIcon.textContent = 'progress_activity';
            statusIcon.className = 'material-symbols-outlined text-primary text-lg step-icon animate-spin';
            return;
        }
        if (state === 'failed') {
            iconBg.className = 'w-10 h-10 rounded-2xl bg-red-50 flex items-center justify-center shrink-0';
            iconEl.className = 'material-symbols-outlined text-red-500 text-lg';
            statusIcon.textContent = 'error';
            statusIcon.className = 'material-symbols-outlined text-red-500 text-lg step-icon';
            return;
        }

        if (state === 'unknown') {
            iconBg.className = 'w-10 h-10 rounded-2xl bg-amber-50 flex items-center justify-center shrink-0';
            iconEl.className = 'material-symbols-outlined text-amber-600 text-lg';
            statusIcon.textContent = 'help';
            statusIcon.className = 'material-symbols-outlined text-amber-600 text-lg step-icon';
            return;
        }

        iconBg.className = 'w-10 h-10 rounded-2xl bg-slate-100 flex items-center justify-center shrink-0';
        iconEl.className = 'material-symbols-outlined text-slate-400 text-lg';
        statusIcon.textContent = 'circle';
        statusIcon.className = 'material-symbols-outlined text-slate-300 text-lg step-icon';
    }

    function updateBadge(label, tone = 'idle') {
        const tones = {
            idle: ['bg-slate-100', 'text-slate-500', 'bg-slate-300'],
            active: ['bg-primary/10', 'text-primary', 'bg-primary'],
            success: ['bg-green-50', 'text-green-700', 'bg-green-500'],
            warning: ['bg-amber-50', 'text-amber-700', 'bg-amber-500'],
            error: ['bg-red-50', 'text-red-700', 'bg-red-500'],
        };
        const [badgeBg, badgeText, dotBg] = tones[tone] || tones.idle;
        analysisBadge.className = `inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-semibold ${badgeBg} ${badgeText}`;
        analysisBadge.innerHTML = `<span class="w-2 h-2 rounded-full ${dotBg}"></span>${label}`;
    }

    function appendFeedItem(key, icon, text) {
        if (!analysisFeed || seenFeedKeys.has(key)) return;
        seenFeedKeys.add(key);
        const item = document.createElement('div');
        item.className = 'flex items-start gap-3 text-sm text-slate-500';
        item.innerHTML = `
            <span class="material-symbols-outlined text-base text-primary mt-0.5">${icon}</span>
            <span class="leading-6">${text}</span>
        `;
        analysisFeed.appendChild(item);
    }

    function updateReviewState(status, publishStatus) {
        submitReviewBtn.classList.add('hidden');
        submitReviewBtn.disabled = true;
        if (openAdminReviewBtn) {
            openAdminReviewBtn.classList.add('hidden');
            if (currentCompanyId) {
                openAdminReviewBtn.href = `/admin/companies?company=${encodeURIComponent(currentCompanyId)}`;
            }
        }

        const qualityWarning = status === 'failed'
            ? '但本轮分析未通过质量核验，请先检查失败阶段，不要直接审核通过。'
            : status === 'unknown'
                ? '但知识库质量尚无完整核验回执，不代表全部成果已经完成。'
                : status !== 'completed'
                    ? '本轮分析仍在进行中，发布状态与知识库处理进度分别记录。'
                    : '';

        // Publication is an existing fact, not evidence that the latest run passed.
        if (publishStatus === 'pending_review' || publishStatus === 'published') {
            const published = publishStatus === 'published';
            reviewStatusTitle.textContent = published ? '已发布' : '已提交审核';
            reviewStatusCopy.textContent = (published
                ? '该公司已通过审核并在前台目录中展示。'
                : '资料已经进入后台审核队列，审核与流水线质量是独立状态。') + qualityWarning;
            openAdminReviewBtn?.classList.remove('hidden');
            updateBadge(status === 'failed' ? '质量未通过'
                : status === 'unknown' ? '质量待核验'
                    : published ? '已发布' : '审核中',
            status === 'failed' ? 'error' : status === 'completed' && published ? 'success' : 'warning');
            return;
        }

        if (status === 'dispatch_unknown') {
            reviewStatusTitle.textContent = '任务派发待确认';
            reviewStatusCopy.textContent = '原任务可能已经进入队列，系统仅观察原公司和任务标识；请勿重复提交或创建新任务。';
            openAdminReviewBtn?.classList.remove('hidden');
            updateBadge('派发待确认', 'warning');
            return;
        }
        if (status === 'failed') {
            reviewStatusTitle.textContent = '分析失败';
            reviewStatusCopy.textContent = '本次分析未通过质量核验。请在后台检查失败阶段并重新分析，不会自动提交审核。';
            openAdminReviewBtn?.classList.remove('hidden');
            updateBadge('质量未通过', 'error');
            return;
        }
        if (status === 'unknown') {
            reviewStatusTitle.textContent = '质量待核验';
            reviewStatusCopy.textContent = '历史“已完成”不能替代质量回执。企业资料、图谱和向量成果核验齐全后才能提交审核。';
            openAdminReviewBtn?.classList.remove('hidden');
            updateBadge('质量待核验', 'warning');
            return;
        }
        if (status === 'completed') {
            reviewStatusTitle.textContent = '等待你确认提交';
            reviewStatusCopy.textContent = '企业资料、图谱和向量成果已经核验，但结果仍停留在草稿状态。点击下方按钮后才会进入后台审核。';
            submitReviewBtn.classList.remove('hidden');
            submitReviewBtn.disabled = false;
            updateBadge('待提交', 'active');
            return;
        }

        reviewStatusTitle.textContent = '分析进行中';
        reviewStatusCopy.textContent = '系统正在处理官网内容并构建企业知识库，完成质量核验后才能提交审核。';
        updateBadge('分析中', 'active');
    }

    function renderSelectedPages(pages) {
        const selectedPages = Array.isArray(pages) ? pages.filter(Boolean) : [];
        if (!selectedPages.length) {
            selectedPagesPanel.classList.add('hidden');
            selectedPagesList.innerHTML = '';
            selectedPagesCount.textContent = '0 页';
            return;
        }

        selectedPagesPanel.classList.remove('hidden');
        selectedPagesCount.textContent = `${selectedPages.length} 页`;
        const roleMap = {
            homepage: '主页',
            about: '公司介绍',
            team: '团队页',
            product: '产品页',
            supporting: '补充页',
        };
        selectedPagesList.innerHTML = selectedPages.map((page, index) => {
            const statusText = page.status === 'captured'
                ? '已锁定'
                : page.status === 'failed'
                    ? '抓取失败'
                    : '待抓取';
            const statusClass = page.status === 'failed'
                ? 'text-red-500 bg-red-50'
                : page.status === 'captured'
                    ? 'text-primary bg-primary/10'
                    : 'text-slate-500 bg-slate-100';
            return `
                <article class="rounded-2xl border border-slate-200 bg-slate-50/70 p-4">
                    <div class="flex items-start justify-between gap-4">
                        <div class="min-w-0">
                            <div class="flex items-center gap-2 mb-2">
                                <span class="inline-flex w-7 h-7 rounded-full bg-primary/10 text-primary text-xs font-bold items-center justify-center">${index + 1}</span>
                                <span class="text-xs font-semibold text-slate-400">${roleMap[page.role] || '关键页'}</span>
                            </div>
                            <h3 class="text-lg font-bold text-slate-900 break-all">${page.title || page.url}</h3>
                            <p class="mt-2 text-sm leading-6 text-slate-500">${page.reason || '该页面将参与企业知识库构建。'}</p>
                            <p class="mt-3 text-xs text-slate-400 break-all">${page.url}</p>
                        </div>
                        <span class="shrink-0 rounded-full px-2.5 py-1 text-[11px] font-semibold ${statusClass}">${statusText}</span>
                    </div>
                </article>
            `;
        }).join('');
    }

    function updateDynamicPanels(status, currentActivity, selectedPages) {
        const config = statusMap[status] || statusMap.pending;
        analysisActivity.textContent = currentActivity || config.activity || config.note;
        analysisNote.textContent = config.note;

        const statusFeed = {
            pending: ['queue', 'hourglass_empty', '已创建任务，等待进入官网解析队列。'],
            crawling: ['crawl', 'travel_explore', selectedPages?.length
                ? `AI 已锁定 ${selectedPages.length} 个关键页面，正在抓取内容。`
                : '已抓取官网首页，正在分析一级目录链接与标题。'],
            cleaning: ['clean', 'data_object', '已完成关键页抓取，正在提取企业信息与结构化摘要。'],
            graph_building: ['graph', 'hub', '正在梳理企业实体、产品能力与关系图谱。'],
            vectorizing: ['vector', 'neurology', '正在将知识内容写入语义索引，支持后续检索与推荐。'],
            completed: ['done', 'check_circle', '企业资料、图谱和向量成果已通过存储读回核验。'],
            failed: ['failed', 'error', '本次分析未通过质量核验，请在后台检查失败阶段。'],
            dispatch_unknown: ['dispatch-unknown', 'hourglass_empty', '派发回执尚未确认，仅观察同一任务，不重复派发。'],
            unknown: ['unknown', 'help', '历史完成状态缺少完整质量回执，企业知识库成果待核验。'],
        };
        const entry = statusFeed[status];
        if (entry) {
            appendFeedItem(entry[0], entry[1], entry[2]);
        }
        if (selectedPages?.length) {
            appendFeedItem('selected-pages', 'scan', `AI 已从首页一级目录中选择 ${selectedPages.length} 个高优先级页面进入最终分析。`);
        }

        renderSelectedPages(selectedPages);
    }

    function savedQuality(payload) {
        return payload.pipeline_quality && typeof payload.pipeline_quality === 'object'
            && !Array.isArray(payload.pipeline_quality) ? payload.pipeline_quality : null;
    }

    function qualityStage(quality, stepName) {
        const stages = quality?.stages;
        if (!stages || typeof stages !== 'object' || Array.isArray(stages)) return null;
        return stepName === 'clean' ? stages.clean || stages.profile || null
            : stepName === 'crawl' ? stages.crawl || stages.source || null
                : stages[stepName] || null;
    }

    function stageVerified(quality, name) {
        const stage = qualityStage(quality, name);
        if (!stage || !['complete', 'passed'].includes(stage.status)) return false;
        if (name === 'crawl') return Number(stage.document_count) > 0;
        if (stage.verified !== true) return false;
        if (name === 'clean') return true;
        if (name === 'graph') return Number(stage.entity_count) > 0;
        if (name === 'vector') return Number(stage.vector_count) > 0;
        return false;
    }

    function effectiveStatus(payload, quality) {
        const status = payload.status || 'pending';
        if (payload.pipeline_dispatch?.state === 'unknown' && status === 'pending') return 'dispatch_unknown';
        const stageFailed = stepOrder.some(name => ['failed', 'degraded'].includes(qualityStage(quality, name)?.status));
        if (payload.error || ['failed', 'degraded'].includes(quality?.status)
            || ['failed', 'degraded'].includes(status) || stageFailed) return 'failed';
        if (status !== 'completed') return statusMap[status] ? status : 'unknown';
        // Top-level completion alone cannot prove durable artifact readback.
        const allVerified = stepOrder.every(name => stageVerified(quality, name));
        return ['complete', 'passed'].includes(quality?.status) && allVerified ? 'completed' : 'unknown';
    }

    function applyPipelineStatus(payload) {
        const quality = savedQuality(payload);
        const status = effectiveStatus(payload, quality);
        const config = statusMap[status] || statusMap.unknown;
        const selectedPages = Array.isArray(payload.selected_pages) ? payload.selected_pages : [];
        currentEffectiveStatus = status;
        currentPublishStatus = payload.publish_status || currentPublishStatus;
        if (payload.company_url) {
            currentNormalizedUrl = payload.company_url;
            analysisUrl.textContent = payload.company_url;
        }

        stepOrder.forEach((stepName, index) => {
            const stageStatus = qualityStage(quality, stepName)?.status;
            if (stageVerified(quality, stepName)) {
                setStepState(stepName, 'done');
            } else if (['failed', 'degraded'].includes(stageStatus)) {
                setStepState(stepName, 'failed');
            } else if (stepName === 'crawl' && selectedPages.length
                && selectedPages.every(page => page?.status === 'captured')) {
                // Saved crawl results prove this stage; later status does not.
                setStepState(stepName, 'done');
            } else if (index === config.active && !['failed', 'unknown'].includes(status)) {
                setStepState(stepName, 'active');
            } else {
                setStepState(stepName, ['failed', 'unknown', 'completed', 'dispatch_unknown'].includes(status)
                    || index < config.active ? 'unknown' : 'idle');
            }
        });

        const qualityError = quality?.failure?.code || ['clean', 'graph', 'vector']
            .map(name => qualityStage(quality, name)?.error).find(Boolean);
        const activity = status === 'unknown' ? statusMap.unknown.activity
            : status === 'failed' ? payload.error || qualityError || payload.current_activity
                : payload.current_activity || null;
        updateDynamicPanels(status, activity, selectedPages);
        if (status === 'failed' && (payload.error || qualityError)) {
            analysisNote.textContent += ` 原因：${payload.error || qualityError}`;
        }
        if (status === 'completed' && ['pending_review', 'published'].includes(currentPublishStatus)) {
            analysisNote.textContent = currentPublishStatus === 'published'
                ? '企业资料、图谱和向量成果已核验；该公司已审核发布。'
                : '企业资料、图谱和向量成果已核验；资料当前已提交后台审核，尚未发布。';
        }
        updateReviewState(status, currentPublishStatus);
        if (status !== 'completed') companyPreviewCard.classList.add('hidden');
        return status;
    }

    async function loadCompanyPreview(companyId, payload = {}) {
        if (currentEffectiveStatus !== 'completed') return;
        // Prefer the saved summary from the lightweight poll; never GET a public
        // detail just to discover quality (that endpoint may hydrate with a model).
        if (payload.company_name && payload.company_summary) {
            companyPreviewCard.classList.remove('hidden');
            previewCompanyName.textContent = payload.company_name;
            previewCompanySummary.textContent = payload.company_summary;
            return;
        }
        if (loadedPreviewIds.has(companyId)) return;
        loadedPreviewIds.add(companyId);
        try {
            const data = await request(`/api/companies/${companyId}`);
            if (currentEffectiveStatus !== 'completed') return;
            if (data.url) analysisUrl.textContent = data.url;
            companyPreviewCard.classList.remove('hidden');
            previewCompanyName.textContent = data.name || '企业名称待核验';
            previewCompanySummary.textContent = data.short_description || data.description || '企业摘要尚未提供。';
        } catch (_) {
            // A preview failure cannot prove quality or trigger a repeated hydration.
        }
    }

    async function pollPipeline(companyId, attempt = 0) {
        clearPolling();
        try {
            const status = await request(`/api/companies/${companyId}/pipeline-status`);
            const displayedStatus = applyPipelineStatus(status);
            if (displayedStatus === 'completed') {
                clearPolling();
                await loadCompanyPreview(companyId, status);
                return;
            }
            if (displayedStatus === 'failed' || displayedStatus === 'unknown') {
                clearPolling();
                return;
            }
        } catch (error) {
            if (attempt >= 10) {
                applyPipelineStatus({
                    status: 'failed',
                    current_activity: error.message,
                    publish_status: currentPublishStatus,
                    selected_pages: [],
                });
                return;
            }
        }

        pollTimer = window.setTimeout(() => {
            pollPipeline(companyId, attempt + 1);
        }, 2000);
    }

    async function startOrResumeAnalysis() {
        const routeState = window.GEOrank?.Routes?.readCompanySubmissionState?.() || {};
        const incomingCompanyId = routeState.companyId || '';
        const incomingUrl = normalizeUrlInput(routeState.url || '');

        currentCompanyId = incomingCompanyId;
        currentNormalizedUrl = incomingUrl;

        if (incomingCompanyId) {
            if (analysisUrl) {
                analysisUrl.textContent = incomingUrl || '正在恢复分析记录...';
            }
            updateBadge('恢复中', 'active');
            await pollPipeline(incomingCompanyId);
            return;
        }

        if (!incomingUrl) {
            analysisUrl.textContent = '未提供有效网址';
            updateBadge('参数缺失', 'error');
            analysisNote.textContent = '请从首页重新输入公司官网地址，再开启分析。';
            analysisActivity.textContent = '当前页面缺少可分析的官网地址。';
            reviewStatusTitle.textContent = '等待开始';
            reviewStatusCopy.textContent = '请返回首页，通过“提交公司”入口输入官网地址后开始分析。';
            return;
        }

        analysisUrl.textContent = incomingUrl;
        updateBadge('创建任务', 'active');
        appendFeedItem('startup', 'language', '正在创建官网分析任务。');

        const result = await request('/api/companies/submit', {
            method: 'POST',
            body: JSON.stringify({ url: incomingUrl }),
        });

        currentCompanyId = result.company_id;
        currentNormalizedUrl = result.normalized_url || incomingUrl;
        analysisUrl.textContent = currentNormalizedUrl;
        history.replaceState(
            {},
            '',
            window.GEOrank.Routes.buildCompanySubmission({
                url: currentNormalizedUrl,
                companyId: currentCompanyId,
            })
        );
        if (result.dispatch_state === 'unknown' || result.status === 'dispatch_unknown') {
            appendFeedItem('dispatch-unknown', 'hourglass_empty', '派发回执尚未确认；已保留原任务标识，仅同步进度，不重复创建。');
        } else if (result.resumed) {
            appendFeedItem('resume', 'history', '检测到已有分析草稿，已为你恢复当前进度。');
        } else {
            appendFeedItem('created', 'hourglass_empty', '分析任务创建成功，正在同步实时进度。');
        }
        await pollPipeline(currentCompanyId);
    }

    async function submitForReview() {
        if (!currentCompanyId || currentEffectiveStatus !== 'completed' || submitReviewBtn.disabled) return;
        submitReviewBtn.disabled = true;
        submitReviewBtn.textContent = '提交中...';
        try {
            const result = await request(`/api/companies/${currentCompanyId}/submit-review`, {
                method: 'POST',
            });
            appendFeedItem('review-submitted', 'task_alt', '分析结果已提交后台审核，审核通过后将在前台目录中展示。');
            currentPublishStatus = result.status === 'published' ? 'published' : 'pending_review';
            updateReviewState(currentEffectiveStatus, currentPublishStatus);
        } catch (error) {
            reviewStatusTitle.textContent = '提交审核失败';
            reviewStatusCopy.textContent = error.message || '请稍后重试。';
            submitReviewBtn.disabled = false;
            submitReviewBtn.textContent = '提交审核';
            updateBadge('提交失败', 'error');
        }
    }

    refreshAnalysisBtn?.addEventListener('click', () => {
        if (currentCompanyId) {
            pollPipeline(currentCompanyId);
        }
    });

    submitReviewBtn?.addEventListener('click', submitForReview);

    startOrResumeAnalysis().catch((error) => {
        analysisNote.textContent = error.message || '初始化分析页失败。';
        analysisActivity.textContent = '无法启动公司分析，请返回首页重试。';
        updateBadge('启动失败', 'error');
        reviewStatusTitle.textContent = '无法启动分析';
        reviewStatusCopy.textContent = error.message || '请返回首页重新发起分析。';
    });
})();
