from __future__ import annotations


def render_dashboard_html() -> str:
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>teamAI Console</title>
    <style>
      :root {
        --bg: #0f1720;
        --panel: rgba(15, 23, 32, 0.78);
        --panel-strong: rgba(12, 20, 27, 0.94);
        --line: rgba(255, 255, 255, 0.1);
        --text: #eef4ef;
        --muted: #9eb0aa;
        --accent: #f2b84b;
        --accent-soft: rgba(242, 184, 75, 0.16);
        --teal: #7cd8c8;
        --rose: #ff9f8f;
        --ok: #76d39b;
        --warn: #f2b84b;
        --fail: #ff7d67;
        --shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
      }

      * {
        box-sizing: border-box;
      }

      html,
      body {
        margin: 0;
        min-height: 100%;
        color: var(--text);
        background:
          radial-gradient(circle at top left, rgba(124, 216, 200, 0.16), transparent 36%),
          radial-gradient(circle at top right, rgba(242, 184, 75, 0.18), transparent 34%),
          linear-gradient(145deg, #0b1118 0%, #101824 40%, #14202a 100%);
        font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
      }

      body::before,
      body::after {
        content: "";
        position: fixed;
        z-index: 0;
        inset: auto;
        width: 22rem;
        height: 22rem;
        border-radius: 999px;
        filter: blur(70px);
        opacity: 0.25;
        pointer-events: none;
      }

      body::before {
        top: 10%;
        right: -6rem;
        background: rgba(255, 159, 143, 0.34);
      }

      body::after {
        bottom: -4rem;
        left: -6rem;
        background: rgba(124, 216, 200, 0.28);
      }

      .shell {
        position: relative;
        z-index: 1;
        max-width: 1320px;
        margin: 0 auto;
        padding: 32px 20px 56px;
      }

      .hero,
      .panel {
        backdrop-filter: blur(18px);
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 24px;
        box-shadow: var(--shadow);
      }

      .hero {
        overflow: hidden;
        padding: 28px;
        position: relative;
        animation: rise 420ms ease-out both;
      }

      .hero::after {
        content: "";
        position: absolute;
        inset: auto -10% -30% 30%;
        height: 18rem;
        background: linear-gradient(90deg, transparent, rgba(242, 184, 75, 0.1), transparent);
        transform: rotate(-10deg);
      }

      .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
        padding: 7px 11px;
        border-radius: 999px;
        background: rgba(124, 216, 200, 0.1);
        color: var(--teal);
        font-size: 12px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }

      h1,
      h2,
      h3 {
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
        font-weight: 700;
        letter-spacing: -0.03em;
      }

      h1 {
        font-size: clamp(2.2rem, 4vw, 4rem);
        line-height: 0.98;
        max-width: 9ch;
      }

      .hero p {
        max-width: 62ch;
        margin: 16px 0 0;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.6;
      }

      .hero-meta {
        margin-top: 22px;
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }

      .pill,
      .status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 9px 12px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: var(--text);
        font-size: 0.93rem;
      }

      .status[data-tone="ok"] {
        color: var(--ok);
        background: rgba(118, 211, 155, 0.12);
      }

      .status[data-tone="warn"] {
        color: var(--warn);
        background: rgba(242, 184, 75, 0.14);
      }

      .status[data-tone="fail"] {
        color: var(--fail);
        background: rgba(255, 125, 103, 0.12);
      }

      .tiles {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin-top: 18px;
      }

      .tile {
        padding: 18px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        animation: rise 520ms ease-out both;
      }

      .tile-label {
        color: var(--muted);
        font-size: 0.84rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
      }

      .tile-value {
        margin-top: 8px;
        font-size: 2rem;
        font-weight: 700;
      }

      .layout {
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
        gap: 18px;
        margin-top: 20px;
      }

      .stack {
        display: grid;
        gap: 18px;
      }

      .panel {
        padding: 22px;
        animation: rise 560ms ease-out both;
      }

      .panel-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
      }

      .panel-copy {
        color: var(--muted);
        font-size: 0.94rem;
      }

      .button,
      button {
        appearance: none;
        border: 0;
        cursor: pointer;
        border-radius: 14px;
        padding: 11px 16px;
        background: linear-gradient(135deg, var(--accent) 0%, #ffcf78 100%);
        color: #201507;
        font-weight: 700;
        transition: transform 160ms ease, box-shadow 160ms ease, opacity 160ms ease;
        box-shadow: 0 12px 24px rgba(242, 184, 75, 0.24);
      }

      .button:hover,
      button:hover {
        transform: translateY(-1px);
      }

      .button.secondary {
        background: rgba(255, 255, 255, 0.06);
        color: var(--text);
        box-shadow: none;
        border: 1px solid rgba(255, 255, 255, 0.08);
      }

      .button:disabled,
      button:disabled {
        cursor: not-allowed;
        opacity: 0.45;
        transform: none;
        box-shadow: none;
      }

      .action-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
      }

      .action-button {
        padding: 7px 12px;
        font-size: 0.86rem;
        border-radius: 10px;
        box-shadow: none;
      }

      form {
        display: grid;
        gap: 14px;
      }

      .field-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }

      label {
        display: grid;
        gap: 8px;
        color: var(--muted);
        font-size: 0.92rem;
      }

      input,
      select,
      textarea {
        width: 100%;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 13px 14px;
        color: var(--text);
        background: rgba(5, 11, 17, 0.55);
        font: inherit;
        outline: none;
      }

      input:focus,
      select:focus,
      textarea:focus {
        border-color: rgba(124, 216, 200, 0.7);
        box-shadow: 0 0 0 3px rgba(124, 216, 200, 0.14);
      }

      textarea {
        min-height: 112px;
        resize: vertical;
      }

      .muted {
        color: var(--muted);
      }

      .list {
        display: grid;
        gap: 12px;
      }

      .list-item {
        border-radius: 18px;
        padding: 15px 16px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
      }

      .list-item[data-active="true"] {
        border-color: rgba(124, 216, 200, 0.45);
        background: rgba(124, 216, 200, 0.08);
      }

      .list-title {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 12px;
      }

      .list-title strong {
        font-size: 0.98rem;
      }

      .mono {
        font-family: "SF Mono", "Cascadia Mono", "Menlo", "Consolas", monospace;
        font-size: 0.9rem;
      }

      .diff {
        margin-top: 10px;
        padding: 12px 14px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(5, 11, 17, 0.55);
        color: var(--text);
        white-space: pre;
        overflow-x: auto;
        max-height: 360px;
        line-height: 1.45;
      }

      .diff .diff-add {
        color: var(--ok);
        background: rgba(118, 211, 155, 0.08);
        display: block;
      }

      .diff .diff-del {
        color: var(--fail);
        background: rgba(255, 125, 103, 0.08);
        display: block;
      }

      .diff .diff-hunk {
        color: var(--teal);
        display: block;
      }

      .diff .diff-meta {
        color: var(--muted);
        display: block;
      }

      .meta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px 10px;
        margin-top: 9px;
        color: var(--muted);
        font-size: 0.87rem;
      }

      .tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 9px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.05);
      }

      .empty {
        padding: 18px;
        border-radius: 18px;
        border: 1px dashed rgba(255, 255, 255, 0.14);
        color: var(--muted);
      }

      .two-up {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 18px;
      }

      .events {
        display: grid;
        gap: 10px;
        max-height: 420px;
        overflow: auto;
        padding-right: 4px;
      }

      .event {
        border-left: 3px solid rgba(124, 216, 200, 0.55);
        padding: 10px 0 10px 12px;
      }

      .event small {
        color: var(--muted);
        display: block;
        margin-bottom: 4px;
      }

      .banner {
        margin-top: 12px;
        padding: 12px 14px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.05);
        color: var(--text);
      }

      .banner[data-tone="ok"] {
        background: rgba(118, 211, 155, 0.12);
        color: var(--ok);
      }

      .banner[data-tone="fail"] {
        background: rgba(255, 125, 103, 0.12);
        color: var(--fail);
      }

      a {
        color: var(--teal);
      }

      @keyframes rise {
        from {
          opacity: 0;
          transform: translateY(10px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      @media (max-width: 1080px) {
        .tiles {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .layout,
        .two-up {
          grid-template-columns: 1fr;
        }
      }

      @media (max-width: 720px) {
        .shell {
          padding: 18px 14px 32px;
        }

        .hero,
        .panel {
          padding: 18px;
          border-radius: 20px;
        }

        .tiles,
        .field-grid {
          grid-template-columns: 1fr;
        }

        h1 {
          max-width: none;
        }
      }
    </style>
  </head>
  <body>
    <main class="shell">
      <section class="hero">
        <div class="eyebrow">teamAI workspace console</div>
        <h1>Run your local agent system from one place.</h1>
        <p>
          teamAI Console brings runtime health, task execution, automation, team plans,
          and agent conversations into a single workspace surface so you can launch work,
          monitor progress, and steer the system without bouncing between raw endpoints.
        </p>
        <div class="hero-meta" id="hero-meta">
          <span class="pill mono" id="hero-model">Loading model...</span>
          <span class="pill mono" id="hero-workspace">Loading workspace...</span>
          <span class="status" id="hero-status" data-tone="warn">Connecting...</span>
          <span class="status" id="hero-writes" data-tone="ok" role="button" tabindex="0" style="cursor:pointer" title="Click to toggle write safety posture (reverts on daemon restart)">writes: read-only</span>
        </div>
        <div class="tiles">
          <div class="tile">
            <div class="tile-label">Recent Runs</div>
            <div class="tile-value" id="stat-jobs">0</div>
          </div>
          <div class="tile">
            <div class="tile-label">Running Now</div>
            <div class="tile-value" id="stat-running">0</div>
          </div>
          <div class="tile">
            <div class="tile-label">Pending Approvals</div>
            <div class="tile-value" id="stat-approvals">0</div>
          </div>
          <div class="tile">
            <div class="tile-label">Automation</div>
            <div class="tile-value" id="stat-schedules">0</div>
          </div>
        </div>
      </section>

      <section class="layout">
        <div class="stack">
          <section class="panel">
            <div class="panel-head">
              <div>
                <h2>New Run</h2>
                <div class="panel-copy">Submit a task directly from the dashboard.</div>
              </div>
              <button class="button secondary" id="refresh-button" type="button">Refresh</button>
            </div>
            <form id="run-form">
              <label>
                Task
                <textarea id="task-input" name="task" placeholder="Inspect the repo and propose the next engineering step." required></textarea>
              </label>
              <div class="field-grid">
                <label>
                  Workspace
                  <input id="workspace-input" name="workspace_path" placeholder="." />
                </label>
                <label>
                  Execution Mode
                  <select id="execution-mode-input" name="execution_mode">
                    <option value="read_only">read_only</option>
                    <option value="workspace_write">workspace_write</option>
                  </select>
                </label>
              </div>
              <div class="field-grid">
                <label>
                  Max Rounds
                  <input id="max-rounds-input" name="max_rounds" type="number" min="1" placeholder="Use default" />
                </label>
                <label>
                  Temperature
                  <input id="temperature-input" name="temperature" type="number" step="0.1" min="0" max="2" placeholder="Use default" />
                </label>
              </div>
              <div style="display:flex; gap:10px; flex-wrap:wrap;">
                <button type="submit">Start Run</button>
                <span class="muted">Runs are queued through the local background job service.</span>
              </div>
            </form>
            <div class="banner" id="submit-banner">Ready for the next task.</div>
          </section>

          <section class="panel">
            <div class="panel-head">
              <div>
                <h2>Recent Runs</h2>
                <div class="panel-copy">Select a run to inspect its execution timeline.</div>
              </div>
            </div>
            <div class="list" id="jobs-list">
              <div class="empty">No runs yet.</div>
            </div>
          </section>

          <section class="panel">
            <div class="panel-head">
              <div>
                <h2>Run Activity</h2>
                <div class="panel-copy">Execution events for the selected run.</div>
              </div>
              <div style="display:flex; gap:8px; align-items:center;">
                <button class="button secondary" id="job-cancel-button" type="button" hidden>Cancel</button>
                <button class="button secondary" id="job-log-button" type="button" hidden>Open Log</button>
                <div class="pill mono" id="selected-job-pill">No run selected</div>
              </div>
            </div>
            <div class="events" id="events-list">
              <div class="empty">Select a run to inspect its execution events.</div>
            </div>
          </section>
        </div>

        <div class="stack">
          <section class="panel">
            <div class="panel-head">
              <div>
                <h2>Pending Approvals</h2>
                <div class="panel-copy">Proposed patches waiting for human review before any file is written.</div>
              </div>
              <button class="button secondary" id="prune-stale-button" type="button">Prune Stale</button>
            </div>
            <div class="list" id="approvals-list">
              <div class="empty">No approvals pending. Patch proposals will appear here for review before any file is written.</div>
            </div>
            <div class="banner" id="approvals-banner">Approve or reject proposed patches here.</div>
          </section>

          <section class="panel">
            <div class="panel-head">
              <div>
                <h2>Automation</h2>
                <div class="panel-copy">Recurring tasks currently configured for this workspace.</div>
              </div>
              <button class="button secondary" id="schedule-new-button" type="button">Add Schedule</button>
            </div>
            <form id="schedule-form" hidden>
              <input type="hidden" name="id" />
              <label>
                Task
                <input name="task" type="text" required placeholder="e.g. Run nightly evals" />
              </label>
              <div class="field-grid">
                <label>
                  Cron
                  <input name="cron" type="text" required placeholder="m h dom mon dow (e.g. 0 3 * * *)" />
                </label>
                <label>
                  Execution mode
                  <select name="execution_mode">
                    <option value="read_only">read_only</option>
                    <option value="workspace_write">workspace_write</option>
                  </select>
                </label>
                <label>
                  Workspace (optional)
                  <input name="workspace" type="text" placeholder="Path. Leave blank for default." />
                </label>
                <label>
                  Max rounds (optional)
                  <input name="max_rounds" type="number" min="1" step="1" />
                </label>
              </div>
              <label style="flex-direction: row; align-items: center; gap: 10px;">
                <input name="enabled" type="checkbox" checked style="width:auto;" />
                Enabled
              </label>
              <div class="action-row">
                <button class="button action-button" type="submit" id="schedule-submit-button">Save Schedule</button>
                <button class="button secondary action-button" type="button" id="schedule-cancel-button">Cancel</button>
              </div>
            </form>
            <div class="list" id="schedules-list">
              <div class="empty">No automation configured. Click <span class="mono">Add Schedule</span> or POST /v1/schedules.</div>
            </div>
            <div class="banner" id="schedules-banner">Add, toggle, edit, or delete scheduled automation here.</div>
          </section>

          <section class="panel">
            <div class="panel-head">
              <div>
                <h2>Team Plans</h2>
                <div class="panel-copy">Multi-agent plans and current execution state.</div>
              </div>
              <button class="button secondary" id="team-new-button" type="button">Launch Team</button>
            </div>
            <form id="team-form" hidden>
              <label>
                Goal
                <input name="goal" type="text" required placeholder="e.g. Audit the supervisor for missing test coverage" />
              </label>
              <label>
                Workspace (optional)
                <input name="workspace_path" type="text" placeholder="Path. Leave blank for default." />
              </label>
              <div class="action-row">
                <button class="button action-button" type="submit" id="team-submit-button">Launch</button>
                <button class="button secondary action-button" type="button" id="team-cancel-button">Cancel</button>
              </div>
            </form>
            <div class="list" id="teams-list">
              <div class="empty">No active team plans. Click <span class="mono">Launch Team</span> or POST /v1/team.</div>
            </div>
            <div class="banner" id="teams-banner">Launch a multi-agent plan here. Decomposition runs in the background.</div>
          </section>

          <section class="panel">
            <div class="panel-head">
              <div>
                <h2>Conversations</h2>
                <div class="panel-copy">Persistent task channels shared across agents in the workspace.</div>
              </div>
            </div>
            <div class="list" id="conversations-list">
              <div class="empty">No conversations captured yet.</div>
            </div>
          </section>
        </div>
      </section>
    </main>

    <script>
      const state = {
        selectedJobId: null,
        loadingEventsFor: null,
        writesEnabled: false,
        schedules: [],
        approvalDiffs: {},
        openApprovalDiffs: new Set(),
        recentJobs: [],
        transcripts: {},
        selectedJobTranscriptOpen: false,
      };

      function escapeHtml(value) {
        return String(value ?? "")
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#39;");
      }

      function formatDate(value) {
        if (!value) {
          return "Unknown";
        }
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) {
          return String(value);
        }
        return date.toLocaleString();
      }

      function statusTone(status) {
        if (status === "completed" || status === "ok" || status === "healthy") {
          return "ok";
        }
        if (status === "failed") {
          return "fail";
        }
        return "warn";
      }

      function statusMarkup(status) {
        const tone = statusTone(status);
        return `<span class="status" data-tone="${tone}">${escapeHtml(status)}</span>`;
      }

      async function fetchJson(url, options = {}) {
        const response = await fetch(url, options);
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || `Request failed: ${response.status}`);
        }
        return response.json();
      }

      function renderJobs(items) {
        const container = document.getElementById("jobs-list");
        if (!items.length) {
          container.innerHTML = '<div class="empty">No runs yet. Submit a task above to see it appear here.</div>';
          return;
        }
        container.innerHTML = items.map((job) => {
          const active = job.job_id === state.selectedJobId;
          const stopReason = job.result?.stop_reason ? `<span class="tag mono">${escapeHtml(job.result.stop_reason)}</span>` : "";
          const taskRoute = job.result?.task_route ? `<span class="tag mono">${escapeHtml(job.result.task_route)}</span>` : "";
          const mode = job.request?.execution_mode ? `<span class="tag mono">${escapeHtml(job.request.execution_mode)}</span>` : "";
          return `
            <button
              type="button"
              class="list-item"
              data-job-id="${escapeHtml(job.job_id)}"
              data-active="${active ? "true" : "false"}"
              style="text-align:left; width:100%;"
            >
              <div class="list-title">
                <strong class="mono">${escapeHtml(job.job_id)}</strong>
                ${statusMarkup(job.status)}
              </div>
              <div class="meta-row">
                <span>${formatDate(job.created_at)}</span>
                ${mode}
                ${taskRoute}
                ${stopReason}
              </div>
            </button>
          `;
        }).join("");
      }

      function renderApprovals(items) {
        const container = document.getElementById("approvals-list");
        if (!items.length) {
          container.innerHTML = '<div class="empty">No approvals pending. Patch proposals will appear here for review before any file is written.</div>';
          return;
        }
        const applyDisabled = !state.writesEnabled;
        const applyTitle = applyDisabled
          ? "Applying approvals is disabled because TEAMAI_ALLOW_WRITES is false."
          : "Apply this approved patch to the workspace.";
        container.innerHTML = items.map((item) => {
          const approvalIdRaw = item.approval_id || "";
          const approvalId = escapeHtml(approvalIdRaw);
          const changeCount = Number(item.change_count || 0);
          const scope = changeCount > 1 ? `${changeCount} files` : item.path || "1 file";
          const tool = item.source_tool ? `<span class="tag mono">${escapeHtml(item.source_tool)}</span>` : "";
          const reason = item.reason ? `<div class="meta-row">${escapeHtml(item.reason)}</div>` : "";
          const diffOpen = state.openApprovalDiffs.has(approvalIdRaw);
          const inspectLabel = diffOpen ? "Hide Diff" : "Show Diff";
          const diffPane = diffOpen
            ? `<pre class="mono diff" data-approval-diff-for="${approvalId}">${renderDiffBody(state.approvalDiffs[approvalIdRaw])}</pre>`
            : "";
          return `
            <div class="list-item" data-approval-id="${approvalId}">
              <div class="list-title">
                <strong class="mono">${approvalId}</strong>
                ${statusMarkup("pending")}
              </div>
              <div class="meta-row">
                <span class="mono">${escapeHtml(String(scope))}</span>
                ${tool}
                <span>${formatDate(item.created_at)}</span>
              </div>
              ${reason}
              <div class="action-row">
                <button
                  type="button"
                  class="button action-button"
                  data-approval-action="apply"
                  data-approval-id="${approvalId}"
                  ${applyDisabled ? "disabled" : ""}
                  title="${escapeHtml(applyTitle)}"
                >Apply</button>
                <button
                  type="button"
                  class="button secondary action-button"
                  data-approval-action="reject"
                  data-approval-id="${approvalId}"
                  title="Mark this approval rejected without touching the workspace."
                >Reject</button>
                <button
                  type="button"
                  class="button secondary action-button"
                  data-approval-action="inspect"
                  data-approval-id="${approvalId}"
                  title="Show the proposed unified diff for this patch."
                >${inspectLabel}</button>
              </div>
              ${diffPane}
            </div>
          `;
        }).join("");
      }

      function renderDiffBody(diffText) {
        if (diffText === undefined) {
          return '<span class="diff-meta">Loading diff...</span>';
        }
        const text = String(diffText || "").trim();
        if (!text) {
          return '<span class="diff-meta">(no textual diff)</span>';
        }
        return text.split("\n").map((line) => {
          let className = "diff-meta";
          if (line.startsWith("+++") || line.startsWith("---")) {
            className = "diff-meta";
          } else if (line.startsWith("@@")) {
            className = "diff-hunk";
          } else if (line.startsWith("+")) {
            className = "diff-add";
          } else if (line.startsWith("-")) {
            className = "diff-del";
          }
          return `<span class="${className}">${escapeHtml(line || " ")}</span>`;
        }).join("");
      }

      async function handleApprovalInspect(approvalId) {
        if (!approvalId) {
          return;
        }
        if (state.openApprovalDiffs.has(approvalId)) {
          state.openApprovalDiffs.delete(approvalId);
          await loadSummary();
          return;
        }
        state.openApprovalDiffs.add(approvalId);
        if (state.approvalDiffs[approvalId] === undefined) {
          await loadSummary();
          try {
            const payload = await fetchJson(`/v1/approvals/${encodeURIComponent(approvalId)}`);
            state.approvalDiffs[approvalId] = String(payload.diff || "");
          } catch (error) {
            state.approvalDiffs[approvalId] = "";
            setApprovalsBanner(error.message || String(error), "fail");
          }
        }
        await loadSummary();
      }

      function setApprovalsBanner(message, tone) {
        const banner = document.getElementById("approvals-banner");
        if (!banner) {
          return;
        }
        banner.textContent = message;
        banner.dataset.tone = tone || "";
      }

      async function handleApprovalAction(approvalId, action) {
        if (!approvalId || !action) {
          return;
        }
        if (action === "apply" && !state.writesEnabled) {
          setApprovalsBanner(
            "Applying approvals is disabled because TEAMAI_ALLOW_WRITES is false.",
            "fail",
          );
          return;
        }
        const label = action === "apply" ? "Applying" : "Rejecting";
        setApprovalsBanner(`${label} approval ${approvalId}...`, "");
        try {
          const body = action === "reject" ? { reason: "Rejected from dashboard." } : {};
          await fetchJson(`/v1/approvals/${encodeURIComponent(approvalId)}/${action}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          });
          const verb = action === "apply" ? "applied" : "rejected";
          setApprovalsBanner(`Approval ${approvalId} ${verb}.`, "ok");
        } catch (error) {
          setApprovalsBanner(error.message || String(error), "fail");
        }
        await loadSummary();
      }

      async function handlePruneStale() {
        setApprovalsBanner("Pruning stale approvals...", "");
        try {
          const response = await fetchJson("/v1/approvals/prune-stale", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: "{}",
          });
          const count = Number(response.count || 0);
          if (count === 0) {
            setApprovalsBanner("No stale approvals to prune.", "");
          } else {
            setApprovalsBanner(`Pruned ${count} stale approval${count === 1 ? "" : "s"}.`, "ok");
          }
        } catch (error) {
          setApprovalsBanner(error.message || String(error), "fail");
        }
        await loadSummary();
      }

      function renderSchedules(items) {
        const container = document.getElementById("schedules-list");
        if (!items.length) {
          container.innerHTML = '<div class="empty">No automation configured. Click <span class="mono">Add Schedule</span> or POST /v1/schedules.</div>';
          return;
        }
        const writesDisabled = !state.writesEnabled;
        const writesTitle = writesDisabled
          ? "Managing schedules is disabled because TEAMAI_ALLOW_WRITES is false."
          : "";
        container.innerHTML = items.map((item) => {
          const idRaw = item.id || "";
          const id = escapeHtml(idRaw);
          const enabled = Boolean(item.enabled);
          const toggleLabel = enabled ? "Disable" : "Enable";
          const toggleTitle = writesDisabled ? writesTitle : `${toggleLabel} this schedule.`;
          const editTitle = writesDisabled ? writesTitle : "Edit this schedule's fields.";
          const deleteTitle = writesDisabled ? writesTitle : "Delete this schedule permanently.";
          const workspace = item.workspace ? `<span>workspace: <span class="mono">${escapeHtml(item.workspace)}</span></span>` : "";
          const rounds = item.max_rounds ? `<span>max_rounds: ${escapeHtml(String(item.max_rounds))}</span>` : "";
          return `
            <div class="list-item" data-schedule-id="${id}">
              <div class="list-title">
                <strong class="mono">${id}</strong>
                ${statusMarkup(enabled ? "enabled" : "disabled")}
              </div>
              <div class="meta-row">
                <span class="tag mono">${escapeHtml(item.cron)}</span>
                <span>${escapeHtml(item.execution_mode || "read_only")}</span>
                ${workspace}
                ${rounds}
              </div>
              <div class="meta-row">${escapeHtml(item.task)}</div>
              <div class="action-row">
                <button
                  type="button"
                  class="button action-button"
                  data-schedule-action="toggle"
                  data-schedule-id="${id}"
                  ${writesDisabled ? "disabled" : ""}
                  title="${escapeHtml(toggleTitle)}"
                >${toggleLabel}</button>
                <button
                  type="button"
                  class="button secondary action-button"
                  data-schedule-action="edit"
                  data-schedule-id="${id}"
                  ${writesDisabled ? "disabled" : ""}
                  title="${escapeHtml(editTitle)}"
                >Edit</button>
                <button
                  type="button"
                  class="button secondary action-button"
                  data-schedule-action="delete"
                  data-schedule-id="${id}"
                  ${writesDisabled ? "disabled" : ""}
                  title="${escapeHtml(deleteTitle)}"
                >Delete</button>
              </div>
            </div>
          `;
        }).join("");
      }

      function setSchedulesBanner(message, tone) {
        const banner = document.getElementById("schedules-banner");
        if (!banner) {
          return;
        }
        banner.textContent = message;
        banner.dataset.tone = tone || "";
      }

      function findScheduleById(id) {
        return (state.schedules || []).find((s) => s.id === id) || null;
      }

      function showScheduleForm(schedule) {
        const form = document.getElementById("schedule-form");
        if (!form) {
          return;
        }
        form.elements.id.value = schedule ? schedule.id : "";
        form.elements.task.value = schedule ? schedule.task || "" : "";
        form.elements.cron.value = schedule ? schedule.cron || "" : "";
        form.elements.execution_mode.value = (schedule && schedule.execution_mode) || "read_only";
        form.elements.workspace.value = schedule ? schedule.workspace || "" : "";
        form.elements.max_rounds.value = schedule && schedule.max_rounds ? String(schedule.max_rounds) : "";
        form.elements.enabled.checked = schedule ? Boolean(schedule.enabled) : true;
        form.hidden = false;
        const submitButton = document.getElementById("schedule-submit-button");
        if (submitButton) {
          submitButton.textContent = schedule ? "Update Schedule" : "Save Schedule";
        }
        form.elements.task.focus();
      }

      function hideScheduleForm() {
        const form = document.getElementById("schedule-form");
        if (!form) {
          return;
        }
        form.reset();
        form.elements.id.value = "";
        form.hidden = true;
      }

      async function submitScheduleForm(event) {
        event.preventDefault();
        if (!state.writesEnabled) {
          setSchedulesBanner(
            "Managing schedules is disabled because TEAMAI_ALLOW_WRITES is false.",
            "fail",
          );
          return;
        }
        const form = event.currentTarget;
        const task = form.elements.task.value.trim();
        const cron = form.elements.cron.value.trim();
        if (!task || !cron) {
          setSchedulesBanner("Task and cron are required.", "fail");
          return;
        }
        const payload = {
          task,
          cron,
          execution_mode: form.elements.execution_mode.value || "read_only",
          enabled: form.elements.enabled.checked,
        };
        const idValue = form.elements.id.value.trim();
        if (idValue) {
          payload.id = idValue;
        }
        const workspaceValue = form.elements.workspace.value.trim();
        if (workspaceValue) {
          payload.workspace = workspaceValue;
        }
        const maxRoundsValue = form.elements.max_rounds.value.trim();
        if (maxRoundsValue) {
          payload.max_rounds = Number(maxRoundsValue);
        }
        const verb = idValue ? "Updating" : "Creating";
        setSchedulesBanner(`${verb} schedule...`, "");
        try {
          const response = await fetchJson("/v1/schedules", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          const finished = idValue ? "updated" : "created";
          setSchedulesBanner(`Schedule ${response.id} ${finished}.`, "ok");
          hideScheduleForm();
        } catch (error) {
          setSchedulesBanner(error.message || String(error), "fail");
        }
        await loadSummary();
      }

      async function handleScheduleAction(scheduleId, action) {
        if (!scheduleId || !action) {
          return;
        }
        if (!state.writesEnabled) {
          setSchedulesBanner(
            "Managing schedules is disabled because TEAMAI_ALLOW_WRITES is false.",
            "fail",
          );
          return;
        }
        if (action === "edit") {
          const schedule = findScheduleById(scheduleId);
          if (!schedule) {
            setSchedulesBanner(`Schedule ${scheduleId} not found.`, "fail");
            return;
          }
          showScheduleForm(schedule);
          setSchedulesBanner(`Editing ${scheduleId}.`, "");
          return;
        }
        if (action === "delete") {
          if (!window.confirm(`Delete schedule ${scheduleId}?`)) {
            return;
          }
          setSchedulesBanner(`Deleting schedule ${scheduleId}...`, "");
          try {
            await fetchJson(`/v1/schedules/${encodeURIComponent(scheduleId)}`, {
              method: "DELETE",
            });
            setSchedulesBanner(`Schedule ${scheduleId} deleted.`, "ok");
          } catch (error) {
            setSchedulesBanner(error.message || String(error), "fail");
          }
          await loadSummary();
          return;
        }
        if (action === "toggle") {
          setSchedulesBanner(`Toggling schedule ${scheduleId}...`, "");
          try {
            const response = await fetchJson(
              `/v1/schedules/${encodeURIComponent(scheduleId)}/toggle`,
              { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" },
            );
            const newState = response.schedule && response.schedule.enabled ? "enabled" : "disabled";
            setSchedulesBanner(`Schedule ${scheduleId} ${newState}.`, "ok");
          } catch (error) {
            setSchedulesBanner(error.message || String(error), "fail");
          }
          await loadSummary();
        }
      }

      function setTeamsBanner(message, tone) {
        const banner = document.getElementById("teams-banner");
        if (!banner) {
          return;
        }
        banner.textContent = message;
        banner.dataset.tone = tone || "";
      }

      function showTeamForm() {
        const form = document.getElementById("team-form");
        if (!form) {
          return;
        }
        form.reset();
        form.hidden = false;
        form.elements.goal.focus();
      }

      function hideTeamForm() {
        const form = document.getElementById("team-form");
        if (!form) {
          return;
        }
        form.reset();
        form.hidden = true;
      }

      async function submitTeamForm(event) {
        event.preventDefault();
        const form = event.currentTarget;
        const goal = form.elements.goal.value.trim();
        if (!goal) {
          setTeamsBanner("A goal is required.", "fail");
          return;
        }
        const payload = { goal };
        const workspaceValue = form.elements.workspace_path.value.trim();
        if (workspaceValue) {
          payload.workspace_path = workspaceValue;
        }
        setTeamsBanner("Launching team plan...", "");
        try {
          const response = await fetchJson("/v1/team", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          setTeamsBanner(`Team ${response.team_id} launched (${response.tasks ? response.tasks.length : 0} tasks).`, "ok");
          hideTeamForm();
        } catch (error) {
          setTeamsBanner(error.message || String(error), "fail");
        }
        await loadSummary();
      }

      function renderTeams(items) {
        const container = document.getElementById("teams-list");
        if (!items.length) {
          container.innerHTML = '<div class="empty">No active team plans. Click <span class="mono">Launch Team</span> or POST /v1/team.</div>';
          return;
        }
        container.innerHTML = items.map((item) => `
          <div class="list-item">
            <div class="list-title">
              <strong>${escapeHtml(item.goal)}</strong>
              ${statusMarkup(item.status)}
            </div>
            <div class="meta-row">
              <span class="tag mono">${escapeHtml(item.team_id)}</span>
              <span>${escapeHtml(String(item.task_count))} tasks</span>
              <span>${escapeHtml(String(item.completed_tasks))} completed</span>
            </div>
          </div>
        `).join("");
      }

      function renderConversations(items) {
        const container = document.getElementById("conversations-list");
        if (!items.length) {
          container.innerHTML = '<div class="empty">No conversations captured yet.</div>';
          return;
        }
        container.innerHTML = items.map((item) => `
          <div class="list-item">
            <div class="list-title">
              <strong>${escapeHtml(item.task_id)}</strong>
              <span class="tag">${escapeHtml(String(item.turn_count))} turns</span>
            </div>
            <div class="meta-row">
              <span>${escapeHtml((item.agents || []).join(", ") || "No agents recorded")}</span>
            </div>
            <div class="meta-row">${formatDate(item.latest_timestamp)}</div>
          </div>
        `).join("");
      }

      async function loadJobEvents(jobId) {
        const container = document.getElementById("events-list");
        if (!jobId) {
          document.getElementById("selected-job-pill").textContent = "No run selected";
          container.innerHTML = '<div class="empty">Select a run to inspect its execution events.</div>';
          return;
        }
        state.loadingEventsFor = jobId;
        document.getElementById("selected-job-pill").textContent = jobId;
        if (state.selectedJobTranscriptOpen && state.transcripts[jobId] !== undefined) {
          const body = state.transcripts[jobId] || "(empty transcript)";
          container.innerHTML = `<pre class="mono diff" style="max-height:480px;">${escapeHtml(body)}</pre>`;
          return;
        }
        try {
          const items = await fetchJson(`/v1/jobs/${encodeURIComponent(jobId)}/events`);
          if (state.loadingEventsFor !== jobId) {
            return;
          }
          if (!items.length) {
            container.innerHTML = '<div class="empty">No events recorded for this run yet.</div>';
            return;
          }
          container.innerHTML = items.slice(-12).reverse().map((event) => `
            <div class="event">
              <small>${escapeHtml(event.kind)} - ${formatDate(event.timestamp)}</small>
              <div>${escapeHtml(event.message)}</div>
            </div>
          `).join("");
        } catch (error) {
          container.innerHTML = `<div class="empty">${escapeHtml(error.message || error)}</div>`;
        }
      }

      function updateJobInspectorControls() {
        const cancelBtn = document.getElementById("job-cancel-button");
        const logBtn = document.getElementById("job-log-button");
        const selected = state.recentJobs.find((job) => job.job_id === state.selectedJobId);
        if (!selected) {
          cancelBtn.hidden = true;
          logBtn.hidden = true;
          return;
        }
        const cancelable = selected.status === "queued" || selected.status === "running";
        cancelBtn.hidden = !cancelable;
        const hasTranscript = selected.status === "completed";
        logBtn.hidden = !hasTranscript;
        logBtn.textContent = state.selectedJobTranscriptOpen ? "Hide Log" : "Open Log";
      }

      async function handleCancelJob() {
        const jobId = state.selectedJobId;
        if (!jobId) return;
        const confirmed = window.confirm(
          `Cancel run ${jobId}?\n\nThe worker keeps executing, but its result will be discarded once the record is terminal.`
        );
        if (!confirmed) return;
        try {
          await fetchJson(`/v1/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
          await loadSummary();
        } catch (error) {
          document.getElementById("events-list").innerHTML =
            `<div class="empty">Cancel failed: ${escapeHtml(error.message || error)}</div>`;
        }
      }

      async function handleToggleLog() {
        const jobId = state.selectedJobId;
        if (!jobId) return;
        state.selectedJobTranscriptOpen = !state.selectedJobTranscriptOpen;
        if (state.selectedJobTranscriptOpen && state.transcripts[jobId] === undefined) {
          try {
            const body = await fetchJson(`/v1/jobs/${encodeURIComponent(jobId)}/transcript`);
            state.transcripts[jobId] = body.transcript || "";
          } catch (error) {
            state.transcripts[jobId] = `Failed to load transcript: ${error.message || error}`;
          }
        }
        updateJobInspectorControls();
        await loadJobEvents(jobId);
      }

      async function loadSummary() {
        try {
          const payload = await fetchJson("/v1/dashboard/summary?limit=8");
          const health = payload.health || {};
          const safety = payload.safety || {};
          const jobs = payload.jobs || { recent: [], counts: {} };
          const schedules = payload.schedules || { items: [] };
          const teams = payload.teams || { items: [] };
          const conversations = payload.conversations || { items: [] };
          const approvals = payload.approvals || { items: [], count: 0 };

          document.getElementById("hero-model").textContent = health.model_id || "Unknown model";
          document.getElementById("hero-workspace").textContent = health.workspace_root || "Unknown workspace";
          const statusElement = document.getElementById("hero-status");
          statusElement.textContent = health.status || "unknown";
          statusElement.dataset.tone = statusTone(health.status);

          const writesElement = document.getElementById("hero-writes");
          const writesEnabled = Boolean(safety.allow_writes);
          state.writesEnabled = writesEnabled;
          writesElement.textContent = writesEnabled ? "writes: enabled" : "writes: read-only";
          writesElement.dataset.tone = writesEnabled ? "warn" : "ok";
          writesElement.title = writesEnabled
            ? "Writes enabled — approved patches can be applied. Click to disable (reverts on daemon restart)."
            : "Writes disabled — runs stay read-only. Click to enable (reverts on daemon restart).";

          document.getElementById("stat-jobs").textContent = String((jobs.recent || []).length);
          document.getElementById("stat-running").textContent = String((jobs.counts || {}).running || 0);
          document.getElementById("stat-schedules").textContent = String((schedules.items || []).length);
          document.getElementById("stat-approvals").textContent = String(approvals.count || (approvals.items || []).length || 0);

          state.recentJobs = jobs.recent || [];
          if (!state.selectedJobId && state.recentJobs.length) {
            state.selectedJobId = state.recentJobs[0].job_id;
          }
          updateJobInspectorControls();

          state.schedules = schedules.items || [];

          const scheduleNewButton = document.getElementById("schedule-new-button");
          if (scheduleNewButton) {
            scheduleNewButton.disabled = !writesEnabled;
            scheduleNewButton.title = writesEnabled
              ? "Create a new scheduled automation."
              : "Managing schedules is disabled because TEAMAI_ALLOW_WRITES is false.";
          }

          renderJobs(jobs.recent || []);
          renderApprovals(approvals.items || []);
          renderSchedules(state.schedules);
          renderTeams(teams.items || []);
          renderConversations(conversations.items || []);
          await loadJobEvents(state.selectedJobId);
        } catch (error) {
          const statusElement = document.getElementById("hero-status");
          statusElement.textContent = "offline";
          statusElement.dataset.tone = "fail";
          document.getElementById("submit-banner").textContent = error.message || String(error);
          document.getElementById("submit-banner").dataset.tone = "fail";
        }
      }

      async function handleWritesToggle() {
        const nextValue = !state.writesEnabled;
        if (nextValue) {
          const workspace = document.getElementById("hero-workspace").textContent || "the workspace";
          const proceed = window.confirm(
            `This permits approved patches to write to ${workspace}. Continue?`
          );
          if (!proceed) {
            return;
          }
        }
        const banner = document.getElementById("submit-banner");
        banner.textContent = "Updating write safety posture...";
        banner.dataset.tone = "";
        try {
          const body = await fetchJson("/v1/settings/allow_writes", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ allow_writes: nextValue }),
          });
          state.writesEnabled = Boolean(body.allow_writes);
          const writesElement = document.getElementById("hero-writes");
          writesElement.textContent = state.writesEnabled ? "writes: enabled" : "writes: read-only";
          writesElement.dataset.tone = state.writesEnabled ? "warn" : "ok";
          writesElement.title = state.writesEnabled
            ? "Writes enabled — approved patches can be applied. Click to disable (reverts on daemon restart)."
            : "Writes disabled — runs stay read-only. Click to enable (reverts on daemon restart).";
          banner.textContent = state.writesEnabled
            ? "Writes enabled. Approved patches can now be applied."
            : "Writes disabled. Runs stay read-only.";
          banner.dataset.tone = state.writesEnabled ? "warn" : "ok";
          await loadSummary();
        } catch (error) {
          banner.textContent = error.message || String(error);
          banner.dataset.tone = "fail";
        }
      }

      async function submitJob(event) {
        event.preventDefault();
        const task = document.getElementById("task-input").value.trim();
        const workspacePath = document.getElementById("workspace-input").value.trim();
        const executionMode = document.getElementById("execution-mode-input").value;
        const maxRoundsRaw = document.getElementById("max-rounds-input").value.trim();
        const temperatureRaw = document.getElementById("temperature-input").value.trim();

        if (!task) {
          return;
        }

        const payload = {
          task,
          execution_mode: executionMode,
        };
        if (workspacePath) {
          payload.workspace_path = workspacePath;
        }
        if (maxRoundsRaw) {
          payload.max_rounds = Number(maxRoundsRaw);
        }
        if (temperatureRaw) {
          payload.temperature = Number(temperatureRaw);
        }

        const banner = document.getElementById("submit-banner");
        banner.textContent = "Queueing run...";
        banner.dataset.tone = "";

        try {
          const response = await fetchJson("/v1/jobs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          state.selectedJobId = response.job_id;
          banner.textContent = `Run queued as ${response.job_id}`;
          banner.dataset.tone = "ok";
          await loadSummary();
        } catch (error) {
          banner.textContent = error.message || String(error);
          banner.dataset.tone = "fail";
        }
      }

      document.getElementById("run-form").addEventListener("submit", submitJob);
      document.getElementById("refresh-button").addEventListener("click", loadSummary);
      document.getElementById("hero-writes").addEventListener("click", handleWritesToggle);
      document.getElementById("hero-writes").addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          handleWritesToggle();
        }
      });
      document.getElementById("jobs-list").addEventListener("click", (event) => {
        const button = event.target.closest("[data-job-id]");
        if (!button) {
          return;
        }
        if (state.selectedJobId !== button.dataset.jobId) {
          state.selectedJobTranscriptOpen = false;
        }
        state.selectedJobId = button.dataset.jobId;
        loadSummary();
      });
      document.getElementById("job-cancel-button").addEventListener("click", handleCancelJob);
      document.getElementById("job-log-button").addEventListener("click", handleToggleLog);
      document.getElementById("approvals-list").addEventListener("click", (event) => {
        const button = event.target.closest("[data-approval-action]");
        if (!button || button.disabled) {
          return;
        }
        const approvalId = button.dataset.approvalId;
        const action = button.dataset.approvalAction;
        if (action === "inspect") {
          handleApprovalInspect(approvalId);
          return;
        }
        handleApprovalAction(approvalId, action);
      });
      document.getElementById("prune-stale-button").addEventListener("click", handlePruneStale);
      document.getElementById("schedules-list").addEventListener("click", (event) => {
        const button = event.target.closest("[data-schedule-action]");
        if (!button || button.disabled) {
          return;
        }
        const scheduleId = button.dataset.scheduleId;
        const action = button.dataset.scheduleAction;
        handleScheduleAction(scheduleId, action);
      });
      document.getElementById("schedule-new-button").addEventListener("click", () => {
        if (!state.writesEnabled) {
          setSchedulesBanner(
            "Managing schedules is disabled because TEAMAI_ALLOW_WRITES is false.",
            "fail",
          );
          return;
        }
        showScheduleForm(null);
        setSchedulesBanner("Creating a new schedule.", "");
      });
      document.getElementById("schedule-cancel-button").addEventListener("click", () => {
        hideScheduleForm();
        setSchedulesBanner("Schedule form closed.", "");
      });
      document.getElementById("schedule-form").addEventListener("submit", submitScheduleForm);
      document.getElementById("team-new-button").addEventListener("click", () => {
        showTeamForm();
        setTeamsBanner("Describe the goal — the planner will decompose it into subtasks.", "");
      });
      document.getElementById("team-cancel-button").addEventListener("click", () => {
        hideTeamForm();
        setTeamsBanner("Team form closed.", "");
      });
      document.getElementById("team-form").addEventListener("submit", submitTeamForm);

      loadSummary();
      setInterval(loadSummary, 5000);
    </script>
  </body>
</html>
"""
