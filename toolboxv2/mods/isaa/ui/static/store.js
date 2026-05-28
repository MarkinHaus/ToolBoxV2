/**
 * ISAA Store — local state mirror.
 * Frames themselves come from the server replay; we only mirror UI state.
 */
(function () {
  'use strict';

  const LS_PREFIX = 'isaa.';

  function lsGet(key, def) {
    try {
      const v = localStorage.getItem(LS_PREFIX + key);
      return v === null ? def : JSON.parse(v);
    } catch (_) {
      return def;
    }
  }
  function lsSet(key, val) {
    try {
      localStorage.setItem(LS_PREFIX + key, JSON.stringify(val));
    } catch (_) {}
  }
  function lsDel(key) {
    try { localStorage.removeItem(LS_PREFIX + key); } catch (_) {}
  }

  const listeners = new Map(); // event -> Set<fn>

  const Store = {
    // session
    activeChatId: lsGet('active_chat_id', null),
    sidebarPanel: lsGet('sidebar_panel', null),
    chats: [],           // [{chat_id, title, agent, last_update, ...}]
    agents: [],          // [{name, model, ...}]
    defaultAgent: 'self',

    // current chat
    chatMeta: null,
    frames: [],          // all frames received from server (canonical)
    lastSeq: 0,
    isRunning: false,
    runId: null,
    attachments: [],     // [{upload_id, vfs_path, name}]

    // ui per chat
    expandedSteps: new Set(),
    l2Steps: new Set(),

    // ----- emit/on -----
    on(ev, fn) {
      if (!listeners.has(ev)) listeners.set(ev, new Set());
      listeners.get(ev).add(fn);
      return () => listeners.get(ev).delete(fn);
    },
    emit(ev, data) {
      const s = listeners.get(ev);
      if (s) s.forEach((fn) => { try { fn(data); } catch (e) { console.error(e); } });
    },

    // ----- chat selection -----
    setActiveChat(id) {
      this.activeChatId = id;
      lsSet('active_chat_id', id);
      this.frames = [];
      this.lastSeq = 0;
      this.chatMeta = null;
      this.runId = null;
      this.isRunning = false;
      this.attachments = [];
      this._loadExpanded();
      this.emit('chat:changed', id);
    },
    _loadExpanded() {
      const id = this.activeChatId;
      if (!id) {
        this.expandedSteps = new Set();
        this.l2Steps = new Set();
        return;
      }
      this.expandedSteps = new Set(lsGet(`expanded.${id}`, []));
      this.l2Steps = new Set(lsGet(`l2.${id}`, []));
    },
    toggleExpanded(stepId) {
      if (this.expandedSteps.has(stepId)) this.expandedSteps.delete(stepId);
      else this.expandedSteps.add(stepId);
      lsSet(`expanded.${this.activeChatId}`, Array.from(this.expandedSteps));
      this.emit('step:toggled', stepId);
    },
    toggleL2(stepId) {
      if (this.l2Steps.has(stepId)) this.l2Steps.delete(stepId);
      else this.l2Steps.add(stepId);
      lsSet(`l2.${this.activeChatId}`, Array.from(this.l2Steps));
      this.emit('step:l2', stepId);
    },

    // ----- sidebar -----
    setPanel(p) {
      this.sidebarPanel = p;
      if (p) lsSet('sidebar_panel', p);
      else lsDel('sidebar_panel');
      this.emit('sidebar:changed', p);
    },

    // ----- frames -----
    pushFrame(frame) {
      // Drop the optimistic (client-only) user_msg once the server's confirmed
      // copy (carrying a seq) arrives, to avoid a duplicate user bubble.
      if (frame.type === 'user_msg' && frame.seq) {
        const i = this.frames.findIndex(f => f.type === 'user_msg' && f._optimistic && f.text === frame.text);
        if (i !== -1) this.frames.splice(i, 1);
      }
      this.frames.push(frame);
      if (frame.seq && frame.seq > this.lastSeq) this.lastSeq = frame.seq;
      // Side effects
      if (frame.type === 'status' || frame.type === 'iteration_start') {
        this.isRunning = true;
      }
      if (frame.type === 'done' || frame.type === 'max_iterations' || frame.type === 'cancelled' || frame.type === 'error') {
        this.isRunning = false;
      }
      if (frame.run_id) this.runId = frame.run_id;
      this.emit('frame', frame);
    },
    resetFrames(frames) {
      this.frames = frames || [];
      this.lastSeq = this.frames.reduce((m, f) => Math.max(m, f.seq || 0), 0);
      this.isRunning = false;
      this.runId = null;
      // Reapply running detection from tail
      for (let i = this.frames.length - 1; i >= 0; i--) {
        const t = this.frames[i].type;
        if (t === 'done' || t === 'max_iterations' || t === 'cancelled' || t === 'error') break;
        if (t === 'status' || t === 'iteration_start' || t === 'content') {
          this.isRunning = true;
          break;
        }
      }
      this.emit('frames:reset');
    },
    truncateAfter(newLastSeq) {
      this.frames = this.frames.filter(f => f.seq <= newLastSeq);
      this.lastSeq = newLastSeq;
      this.emit('frames:reset');
    },

    // ----- attachments -----
    addAttachment(a) { this.attachments.push(a); this.emit('attachments'); },
    removeAttachment(uploadId) {
      this.attachments = this.attachments.filter(a => a.upload_id !== uploadId);
      this.emit('attachments');
    },
    clearAttachments() { this.attachments = []; this.emit('attachments'); },

    // ----- drafts -----
    getDraft() { return this.activeChatId ? lsGet(`draft.${this.activeChatId}`, '') : ''; },
    setDraft(text) { if (this.activeChatId) lsSet(`draft.${this.activeChatId}`, text); },
    clearDraft() { if (this.activeChatId) lsDel(`draft.${this.activeChatId}`); },

    // ----- scroll -----
    getScroll() { return this.activeChatId ? lsGet(`scroll.${this.activeChatId}`, 0) : 0; },
    setScroll(top) { if (this.activeChatId) lsSet(`scroll.${this.activeChatId}`, top); },
  };

  window.ISAA = window.ISAA || {};
  window.ISAA.Store = Store;
})();
