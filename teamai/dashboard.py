from __future__ import annotations


def render_dashboard_html() -> str:
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>teamAI Control</title>
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
        <div class="eyebrow">teamAI gateway control</div>
        <h1>Turn the loop into a control room.</h1>
        <p>
          OpenClaw feels like a product because it gives you a place to drive the agent:
          health, runs, schedules, teams, and conversations in one browser surface. This
          dashboard gives teamAI that same immediate, operator-friendly entry point.
        </p>
        <div class="hero-meta" id="hero-meta">
          <span class="pill mono" id="hero-model">Loading model...</span>
          <span class="pill mono" id="hero-workspace">Loading workspace...</span>
          <span class="status" id="hero-status" data-tone="warn">Connecting...</span>
        </div>
        <div class="tiles">
          <div class="tile">
            <div class="tile-label">Recent Jobs</div>
            <div class="tile-value" id="stat-jobs">0</div>
          </div>
          <div class="tile">
            <div class="tile-label">Running Now</div>
            <div class="tile-value" id="stat-running">0</div>
          </div>
          <div class="tile">
            <div class="tile-label">Schedules</div>
            <div class="tile-value" id="stat-schedules">0</div>
          </div>
          <div class="tile">
            <div class="tile-label">Conversations</div>
            <div class="tile-value" id="stat-conversations">0</div>
          </div>
        </div>
      </section>

      <section class="layout">
        <div class="stack">
          <section class="panel">
            <div class="panel-head">
              <div>
                <h2>Launch A Run</h2>
                <div class="panel-copy">Queue a task without leaving the browser.</div>
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
                <button type="submit">Queue Job</button>
                <span class="muted">Runs are submitted to the existing background job API.</span>
              </div>
            </form>
            <div class="banner" id="submit-banner">Waiting for your next task.</div>
          </section>

          <section class="panel">
            <div class="panel-head">
              <div>
                <h2>Recent Jobs</h2>
                <div class="panel-copy">Select a run to inspect its live event trail.</div>
              </div>
            </div>
            <div class="list" id="jobs-list">
              <div class="empty">No jobs yet.</div>
            </div>
          </section>

          <section class="panel">
            <div class="panel-head">
              <div>
                <h2>Job Activity</h2>
                <div class="panel-copy">Event stream for the selected run.</div>
              </div>
              <div class="pill mono" id="selected-job-pill">No job selected</div>
            </div>
            <div class="events" id="events-list">
              <div class="empty">Pick a job to inspect its progress events.</div>
            </div>
          </section>
        </div>

        <div class="stack">
          <section class="panel">
            <div class="panel-head">
              <div>
                <h2>Schedules</h2>
                <div class="panel-copy">Recurring prompts currently configured for the gateway.</div>
              </div>
            </div>
            <div class="list" id="schedules-list">
              <div class="empty">No schedules configured.</div>
            </div>
          </section>

          <section class="panel">
            <div class="panel-head">
              <div>
                <h2>Team Runs</h2>
                <div class="panel-copy">Multi-agent plans and current execution state.</div>
              </div>
            </div>
            <div class="list" id="teams-list">
              <div class="empty">No active team plans.</div>
            </div>
          </section>

          <section class="panel">
            <div class="panel-head">
              <div>
                <h2>Conversation Inbox</h2>
                <div class="panel-copy">Persistent inter-agent task channels inside the workspace.</div>
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
          container.innerHTML = '<div class="empty">No jobs yet.</div>';
          return;
        }
        container.innerHTML = items.map((job) => {
          const active = job.job_id === state.selectedJobId;
          const stopReason = job.result?.stop_reason ? `<span class="tag mono">${escapeHtml(job.result.stop_reason)}</span>` : "";
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
                ${stopReason}
              </div>
            </button>
          `;
        }).join("");
      }

      function renderSchedules(items) {
        const container = document.getElementById("schedules-list");
        if (!items.length) {
          container.innerHTML = '<div class="empty">No schedules configured.</div>';
          return;
        }
        container.innerHTML = items.map((item) => `
          <div class="list-item">
            <div class="list-title">
              <strong>${escapeHtml(item.id)}</strong>
              ${statusMarkup(item.enabled ? "enabled" : "disabled")}
            </div>
            <div class="meta-row">
              <span class="tag mono">${escapeHtml(item.cron)}</span>
              <span>${escapeHtml(item.execution_mode || "read_only")}</span>
            </div>
            <div class="meta-row">${escapeHtml(item.task)}</div>
          </div>
        `).join("");
      }

      function renderTeams(items) {
        const container = document.getElementById("teams-list");
        if (!items.length) {
          container.innerHTML = '<div class="empty">No active team plans.</div>';
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
        if (!jobId) {
          document.getElementById("selected-job-pill").textContent = "No job selected";
          document.getElementById("events-list").innerHTML = '<div class="empty">Pick a job to inspect its progress events.</div>';
          return;
        }
        state.loadingEventsFor = jobId;
        document.getElementById("selected-job-pill").textContent = jobId;
        try {
          const items = await fetchJson(`/v1/jobs/${encodeURIComponent(jobId)}/events`);
          if (state.loadingEventsFor !== jobId) {
            return;
          }
          const container = document.getElementById("events-list");
          if (!items.length) {
            container.innerHTML = '<div class="empty">No events recorded for this job yet.</div>';
            return;
          }
          container.innerHTML = items.slice(-12).reverse().map((event) => `
            <div class="event">
              <small>${escapeHtml(event.kind)} - ${formatDate(event.timestamp)}</small>
              <div>${escapeHtml(event.message)}</div>
            </div>
          `).join("");
        } catch (error) {
          document.getElementById("events-list").innerHTML = `<div class="empty">${escapeHtml(error.message || error)}</div>`;
        }
      }

      async function loadSummary() {
        try {
          const payload = await fetchJson("/v1/dashboard/summary?limit=8");
          const health = payload.health || {};
          const jobs = payload.jobs || { recent: [], counts: {} };
          const schedules = payload.schedules || { items: [] };
          const teams = payload.teams || { items: [] };
          const conversations = payload.conversations || { items: [] };

          document.getElementById("hero-model").textContent = health.model_id || "Unknown model";
          document.getElementById("hero-workspace").textContent = health.workspace_root || "Unknown workspace";
          const statusElement = document.getElementById("hero-status");
          statusElement.textContent = health.status || "unknown";
          statusElement.dataset.tone = statusTone(health.status);

          document.getElementById("stat-jobs").textContent = String((jobs.recent || []).length);
          document.getElementById("stat-running").textContent = String((jobs.counts || {}).running || 0);
          document.getElementById("stat-schedules").textContent = String((schedules.items || []).length);
          document.getElementById("stat-conversations").textContent = String((conversations.items || []).length);

          if (!state.selectedJobId && (jobs.recent || []).length) {
            state.selectedJobId = jobs.recent[0].job_id;
          }

          renderJobs(jobs.recent || []);
          renderSchedules(schedules.items || []);
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
        banner.textContent = "Queueing job...";
        banner.dataset.tone = "";

        try {
          const response = await fetchJson("/v1/jobs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          state.selectedJobId = response.job_id;
          banner.textContent = `Queued ${response.job_id}`;
          banner.dataset.tone = "ok";
          await loadSummary();
        } catch (error) {
          banner.textContent = error.message || String(error);
          banner.dataset.tone = "fail";
        }
      }

      document.getElementById("run-form").addEventListener("submit", submitJob);
      document.getElementById("refresh-button").addEventListener("click", loadSummary);
      document.getElementById("jobs-list").addEventListener("click", (event) => {
        const button = event.target.closest("[data-job-id]");
        if (!button) {
          return;
        }
        state.selectedJobId = button.dataset.jobId;
        loadSummary();
      });

      loadSummary();
      setInterval(loadSummary, 5000);
    </script>
  </body>
</html>
"""
