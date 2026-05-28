/**
 * Chat rendering: step cards, tool pills, pretty key/value rendering.
 *
 * Frame types from a_stream + bridge:
 *   user_msg, status, content, reasoning, tool_start, tool_result,
 *   final_answer, paused, cancelled, done, max_iterations, error,
 *   iteration_start, narrator, warning, post_processing, rollback_done
 *
 * Rendering hierarchy:
 *   L0 (always): step header (summary) + body (final content)
 *   L1 (click header): reasoning + tool pills with args+result preview
 *   L2 (click "Details"): everything ungekürzt, raw kv tree
 */
(function () {
  'use strict';

  const HUMAN_TOOL_NAMES = {
    vfs_read: 'Datei lesen',
    vfs_write: 'Datei schreiben',
    vfs_list: 'Verzeichnis listen',
    vfs_ls: 'Verzeichnis listen',
    vfs_create: 'Datei anlegen',
    vfs_delete: 'Datei löschen',
    web_search: 'Web-Suche',
    web_fetch: 'URL abrufen',
    think: 'Nachdenken',
    final_answer: 'Antwort formulieren',
    memory_inject: 'Memory speichern',
    memory_recall: 'Memory abrufen',
    shell: 'Shell ausführen',
    code: 'Code ausführen',
  };

  const TRUNC_STRING = 200;
  const TRUNC_LIST = 5;
  const TRUNC_DICT = 6;
  const TRUNC_PILL_ARG = 80;

  // ============================================================================
  // PRETTY VALUE RENDERING
  // ============================================================================

  function escape(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[c]);
  }

  function isMonoLike(s) {
    if (typeof s !== 'string') return false;
    return /^\/[^\s]+|^https?:\/\/|^[a-f0-9]{8,}-|^run_[a-f0-9]{6,}/i.test(s);
  }

  function renderValueShort(v, maxStr = TRUNC_STRING) {
    if (v === null || v === undefined) return '<span class="v">null</span>';
    if (typeof v === 'string') {
      const cls = isMonoLike(v) ? 'v mono' : 'v';
      const s = v.length > maxStr ? v.slice(0, maxStr) + '…' : v;
      return `<span class="${cls}">${escape(s)}</span>`;
    }
    if (typeof v === 'number' || typeof v === 'boolean') {
      return `<span class="v mono">${escape(String(v))}</span>`;
    }
    if (Array.isArray(v)) {
      if (v.length === 0) return '<span class="v">[]</span>';
      const head = v.slice(0, TRUNC_LIST).map(x => renderValueShort(x, 40)).join(', ');
      const tail = v.length > TRUNC_LIST ? ` … (${v.length - TRUNC_LIST} more)` : '';
      return `<span class="v">[${head}${tail}]</span>`;
    }
    if (typeof v === 'object') {
      const keys = Object.keys(v);
      if (keys.length === 0) return '<span class="v">{}</span>';
      const head = keys.slice(0, 3).map(k => `${escape(k)}: ${renderValueShort(v[k], 40)}`).join(', ');
      const tail = keys.length > 3 ? `, … (${keys.length - 3} more)` : '';
      return `<span class="v">{${head}${tail}}</span>`;
    }
    return `<span class="v">${escape(String(v))}</span>`;
  }

  /** Pretty key=value inline (for tool pill). */
  function renderArgsInline(args, maxLen = TRUNC_PILL_ARG) {
    if (!args || typeof args !== 'object') return '';
    const parts = [];
    let used = 0;
    for (const [k, v] of Object.entries(args)) {
      const vStr = typeof v === 'string' ? v : JSON.stringify(v);
      const truncV = vStr.length > maxLen ? vStr.slice(0, maxLen) + '…' : vStr;
      const part = `<span class="k">${escape(k)}:</span> <span class="v">${escape(truncV)}</span>`;
      parts.push(part);
      used += k.length + truncV.length + 2;
      if (used > maxLen * 2) {
        parts.push('<span class="k">…</span>');
        break;
      }
    }
    return parts.join('&nbsp;&nbsp;');
  }

  /** Full ungekürztes kv-tree für Level 2. */
  function renderKvTree(v, depth = 0) {
    if (v === null || v === undefined) return '<span class="v">null</span>';
    if (typeof v === 'string') {
      if (v.includes('\n') || v.length > 80) {
        return `<pre>${escape(v)}</pre>`;
      }
      return `<span class="v">${escape(v)}</span>`;
    }
    if (typeof v === 'number' || typeof v === 'boolean') {
      return `<span class="v">${escape(String(v))}</span>`;
    }
    if (Array.isArray(v)) {
      if (v.length === 0) return '<span class="v">[]</span>';
      return '<ul>' + v.map((x, i) => `<li><span class="k">[${i}]</span> ${renderKvTree(x, depth + 1)}</li>`).join('') + '</ul>';
    }
    if (typeof v === 'object') {
      const entries = Object.entries(v);
      if (entries.length === 0) return '<span class="v">{}</span>';
      return '<ul>' + entries.map(([k, val]) => `<li><span class="k">${escape(k)}:</span> ${renderKvTree(val, depth + 1)}</li>`).join('') + '</ul>';
    }
    return `<span class="v">${escape(String(v))}</span>`;
  }

  function humanToolName(name) {
    return HUMAN_TOOL_NAMES[name] || name;
  }

  // ============================================================================
  // STEP DOM BUILDING
  // ============================================================================

  /**
   * Grouping rule: a "step" is bounded by `iteration_start` OR a top-level
   * user-message → final_answer/done. We render each step as a card.
   *
   * Within a step, frames accumulate:
   *   content/reasoning chunks → coalesce into a single body/reasoning blob
   *   tool_start → new pill; matching tool_result → fill pill
   *   final_answer → body (overrides content)
   */
  function groupFrames(frames) {
    const groups = []; // array of { user?, steps: [step] }
    let currentTurn = null;
    let currentStep = null;

    const startTurn = () => {
      currentTurn = { user: null, steps: [], terminal: null };
      groups.push(currentTurn);
    };
    const startStep = (stepId, iter) => {
      currentStep = {
        step_id: stepId || (`step:${(currentTurn?.steps.length || 0)}`),
        iter: iter || 0,
        body: '',
        reasoning: '',
        tools: [], // {id, name, args, result, status, raw}
        narrator: [],
        warnings: [],
        error: null,
        finalAnswer: null,
        rawFrames: [],
      };
      if (!currentTurn) startTurn();
      currentTurn.steps.push(currentStep);
    };

    for (const f of frames) {
      if (f.type === 'user_msg') {
        startTurn();
        currentTurn.user = f;
        startStep(`u:${f.seq}`, 0);
        continue;
      }
      if (!currentStep) {
        startTurn();
        startStep(f.step_id, f.iter);
      }
      currentStep.rawFrames.push(f);

      switch (f.type) {
        case 'iteration_start':
          if (currentStep.body || currentStep.tools.length) {
            startStep(f.step_id, f.iter);
            currentStep.rawFrames.push(f);
          } else {
            currentStep.step_id = f.step_id || currentStep.step_id;
            currentStep.iter = f.iter || currentStep.iter;
          }
          break;
        case 'content':
          currentStep.body += (f.chunk || '');
          break;
        case 'reasoning':
          currentStep.reasoning += (f.chunk || '');
          break;
        case 'tool_start': {
          const tid = f.id || `${f.name}-${currentStep.tools.length}`;
          currentStep.tools.push({
            id: tid, name: f.name, args: f.args || {}, result: null,
            status: 'running', raw: { start: f, end: null },
          });
          break;
        }
        case 'tool_result': {
          // Find matching tool by id (fallback to last open by name)
          const tools = currentStep.tools;
          let tool = null;
          if (f.id) tool = tools.find(t => t.id === f.id);
          if (!tool) tool = [...tools].reverse().find(t => t.name === f.name && t.status === 'running');
          if (!tool) {
            tool = { id: f.id || `${f.name}-late`, name: f.name, args: {}, result: null, status: 'ok', raw: { start: null, end: f } };
            tools.push(tool);
          }
          tool.result = f.result;
          tool.status = (f.is_final === false) ? 'running' : (f.error ? 'err' : 'ok');
          tool.raw.end = f;
          break;
        }
        case 'final_answer':
          currentStep.finalAnswer = f.answer || '';
          break;
        case 'narrator':
          if (f.narrator_msg) currentStep.narrator.push(f.narrator_msg);
          break;
        case 'warning':
          if (f.message) currentStep.warnings.push(f.message);
          break;
        case 'error':
          currentStep.error = f.error || 'Unknown error';
          break;
        case 'max_iterations':
        case 'paused':
        case 'cancelled':
        case 'done':
          currentTurn.terminal = f;
          break;
        case 'rollback_done':
          // handled at higher level
          break;
        case 'status':
        case 'post_processing':
        default:
          break;
      }
    }
    return groups;
  }

  // ============================================================================
  // RENDERERS
  // ============================================================================

  function renderUserMsg(f) {
    const attHtml = (f.attachments || []).map(a =>
      `<div class="msg-att">📎 ${escape(a.vfs_path || a.name || '')}</div>`
    ).join('');
    return `<div class="msg msg-user">${escape(f.text || '')}${attHtml}</div>`;
  }

  function renderToolPill(t, stepId, l1Open) {
    const argsInline = renderArgsInline(t.args);
    const statusIcon = t.status === 'running' ? '⟳' : (t.status === 'ok' ? '✓' : '✕');
    const resultPreview = t.result == null
      ? ''
      : (typeof t.result === 'string'
         ? (t.result.length > TRUNC_STRING ? t.result.slice(0, TRUNC_STRING) + '…' : t.result)
         : JSON.stringify(t.result).slice(0, TRUNC_STRING));
    return `
      <div class="tool-pill" data-status="${t.status}" data-tool-id="${escape(t.id)}" data-step-id="${escape(stepId)}" data-l1="${l1Open ? 'true' : 'false'}">
        <span class="tp-icon">⚙</span>
        <span class="tp-name">${escape(humanToolName(t.name))}</span>
        <span class="tp-args">${argsInline}</span>
        <span class="tp-status">${statusIcon}</span>
        ${resultPreview ? `<div class="tp-result">${escape(resultPreview)}</div>` : ''}
      </div>
    `;
  }

  function renderStepL2(step) {
    let html = '<div class="step-l2-section"><h4>Args / Results (full)</h4>';
    if (step.tools.length === 0) {
      html += '<div class="kv-tree"><em>no tools</em></div>';
    } else {
      for (const t of step.tools) {
        html += `<div style="margin-bottom: 12px;">`;
        html += `<div class="k">${escape(t.name)} (status=${escape(t.status)})</div>`;
        html += `<div class="kv-tree">args: ${renderKvTree(t.args)}</div>`;
        html += `<div class="kv-tree">result: ${renderKvTree(t.result)}</div>`;
        html += `</div>`;
      }
    }
    html += '</div>';

    if (step.reasoning) {
      html += '<div class="step-l2-section"><h4>Reasoning (full)</h4>';
      html += `<pre>${escape(step.reasoning)}</pre></div>`;
    }

    if (step.narrator.length) {
      html += '<div class="step-l2-section"><h4>Narrator</h4>';
      html += step.narrator.map(m => `<div class="kv-tree">${escape(m)}</div>`).join('');
      html += '</div>';
    }

    html += '<div class="step-l2-section"><h4>Raw frames</h4>';
    html += `<pre>${escape(step.rawFrames.map(f => JSON.stringify(f)).join('\n'))}</pre></div>`;
    return html;
  }

  function buildHeaderSummary(step) {
    if (step.error) return 'Fehler';
    if (step.tools.length === 0 && !step.body && !step.finalAnswer) return 'Nachgedacht';
    const parts = [];
    if (step.tools.length) {
      const toolNames = step.tools.slice(0, 3).map(t => humanToolName(t.name));
      parts.push(toolNames.join(', '));
      if (step.tools.length > 3) parts.push(`+${step.tools.length - 3}`);
    }
    if (step.body || step.finalAnswer) parts.push('geantwortet');
    return parts.join(' · ');
  }

  function renderStep(step, isLast, expanded, l2) {
    const sid = step.step_id;
    const body = step.finalAnswer != null ? step.finalAnswer : step.body;
    const summary = buildHeaderSummary(step);
    const isThinking = isLast && window.ISAA.Store.isRunning && !step.finalAnswer && !step.error;

    let toolsHtml = '';
    if (step.tools.length) {
      toolsHtml = step.tools.map(t => renderToolPill(t, sid, expanded)).join('');
    }

    let reasoningHtml = '';
    if (step.reasoning) {
      const short = step.reasoning.length > TRUNC_STRING ? step.reasoning.slice(0, TRUNC_STRING) + '…' : step.reasoning;
      reasoningHtml = `<div class="reasoning">${escape(short)}</div>`;
    }

    const warningHtml = step.warnings.length
      ? `<div class="reasoning"><em>${step.warnings.map(w => escape(w)).join(' · ')}</em></div>`
      : '';
    const errorHtml = step.error ? `<div class="step-error">${escape(step.error)}</div>` : '';

    return `
      <div class="step-card${isThinking ? ' is-thinking' : ''}" data-step-id="${escape(sid)}" data-expanded="${expanded ? 'true' : 'false'}" data-l2="${l2 ? 'true' : 'false'}">
        <span class="step-anchor" data-action="rollback" data-step-id="${escape(sid)}" title="Zu diesem Schritt zurück"></span>
        <div class="step-header" data-action="toggle" data-step-id="${escape(sid)}">
          <svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 6l6 6-6 6"/></svg>
          <span>${escape(summary)}</span>
          <span class="thinking">thinking…</span>
        </div>
        ${body ? `<div class="step-body">${escape(body)}</div>` : ''}
        ${errorHtml}
        <div class="step-l1">
          ${warningHtml}
          ${reasoningHtml}
          ${toolsHtml}
        </div>
        <div class="step-l2-link" data-action="toggle-l2" data-step-id="${escape(sid)}">${l2 ? '▾ hide details' : '▸ details'}</div>
        <div class="step-l2">${l2 ? renderStepL2(step) : ''}</div>
      </div>
    `;
  }

  function render(container) {
    const store = window.ISAA.Store;
    const groups = groupFrames(store.frames);
    if (groups.length === 0) {
      container.innerHTML = '';
      return;
    }
    let html = '<div class="chat-stream">';
    for (let gi = 0; gi < groups.length; gi++) {
      const g = groups[gi];
      if (g.user) html += renderUserMsg(g.user);
      for (let si = 0; si < g.steps.length; si++) {
        const step = g.steps[si];
        const isLastOverall = (gi === groups.length - 1) && (si === g.steps.length - 1);
        const hasContent = step.body || step.tools.length || step.reasoning || step.error || step.finalAnswer;
        // Keep the trailing empty step visible while running → immediate thinking feedback.
        if (!hasContent && !(isLastOverall && store.isRunning)) continue;
        const expanded = store.expandedSteps.has(step.step_id);
        const l2 = store.l2Steps.has(step.step_id);
        html += renderStep(step, isLastOverall, expanded, l2);
      }
    }
    html += '</div>';
    container.innerHTML = html;
    // §8: ISA logo spinner pinned at the current streaming position (moves down
    // with the content as new frames arrive + auto-scroll).
    if (store.isRunning) {
      const tpl = document.getElementById('tpl-logo');
      const stream = container.querySelector('.chat-stream');
      if (tpl && stream) {
        const sp = document.createElement('div');
        sp.className = 'stream-spinner';
        sp.appendChild(tpl.content.cloneNode(true));
        stream.appendChild(sp);
      }
    }
  }

  // ============================================================================
  // EVENT BINDING (delegated on container)
  // ============================================================================

  function bindEvents(container) {
    container.addEventListener('click', async (e) => {
      const t = e.target;
      const action = t.closest('[data-action]')?.dataset?.action;
      const stepId = t.closest('[data-step-id]')?.dataset?.stepId;
      if (!action) return;
      if (action === 'toggle' && stepId) {
        e.preventDefault();
        window.ISAA.Store.toggleExpanded(stepId);
      } else if (action === 'toggle-l2' && stepId) {
        e.preventDefault();
        e.stopPropagation();
        window.ISAA.Store.toggleL2(stepId);
      } else if (action === 'rollback' && stepId) {
        e.preventDefault();
        e.stopPropagation();
        if (await window.ISAA.UI.confirm('Zu diesem Schritt zurück? Alles danach wird verworfen.', { danger: true, okLabel: 'Zurücksetzen' })) {
          window.ISAA.WS.send({ op: 'rollback', step_id: stepId });
        }
      }
      // Tool pill click: toggle L1 highlight (its parent step gets expanded too)
      const pill = t.closest('.tool-pill');
      if (pill && !action) {
        const cur = pill.dataset.l1 === 'true';
        pill.dataset.l1 = cur ? 'false' : 'true';
      }
    });
  }

  // ============================================================================
  // SCROLL MGT
  // ============================================================================

  function scrollToBottom(container) {
    const parent = container.parentElement;
    if (!parent) return;
    parent.scrollTop = parent.scrollHeight;
  }

  window.ISAA = window.ISAA || {};
  window.ISAA.Chat = { render, bindEvents, scrollToBottom };
})();
