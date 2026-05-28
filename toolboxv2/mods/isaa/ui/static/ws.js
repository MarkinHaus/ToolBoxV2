/**
 * ISAA WS client — reconnects + sends `hello` with last_seq.
 *
 * The WS uses fixed path /ws/chat. The chat_id flows over the hello frame.
 */
(function () {
  'use strict';

  const RECONNECT_INITIAL = 500;
  const RECONNECT_MAX = 15000;
  const PING_INTERVAL = 20000;

  function url() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    // WS worker runs on its own port (default :8100). Override via window.__TB_WS_PORT__.
    const port = window.__TB_WS_PORT__ || '8100';
    return `${proto}//${location.hostname}:${port}/ws/open_chat`;
  }

  class WS {
    constructor() {
      this.ws = null;
      this.reconnectTimer = null;
      this.reconnectDelay = RECONNECT_INITIAL;
      this.pingTimer = null;
      this.chatId = null;
      this.connected = false;
      this.pending = [];   // queued sends while socket not open (first-message race)
      this.handlers = {
        frame: null,
        open: null,
        close: null,
      };
    }

    setHandlers(h) { this.handlers = Object.assign(this.handlers, h); }

    connect(chatId) {
      this.chatId = chatId;
      this._open();
    }

    _open() {
      if (this.ws && this.ws.readyState <= 1) {
        try { this.ws.close(); } catch (_) {}
      }
      try {
        this.ws = new WebSocket(url());
      } catch (e) {
        console.warn('[ws] open failed', e);
        this._scheduleReconnect();
        return;
      }
      this.ws.onopen = () => {
        this.connected = true;
        this.reconnectDelay = RECONNECT_INITIAL;
        // hello with last_seq from store
        const lastSeq = (window.ISAA && window.ISAA.Store) ? window.ISAA.Store.lastSeq : 0;
        this.send({ op: 'hello', chat_id: this.chatId, last_seq: lastSeq });
        this._startPing();
        // Flush anything queued while the socket was still connecting.
        const q = this.pending; this.pending = [];
        for (const m of q) this.send(m);
        if (this.handlers.open) this.handlers.open();
      };
      this.ws.onmessage = (e) => {
        let data;
        try { data = JSON.parse(e.data); }
        catch (_) { return; }
        if (this.handlers.frame) this.handlers.frame(data);
      };
      this.ws.onerror = () => {};
      this.ws.onclose = () => {
        this.connected = false;
        this._stopPing();
        if (this.handlers.close) this.handlers.close();
        this._scheduleReconnect();
      };
    }

    _scheduleReconnect() {
      if (!this.chatId) return;
      if (this.reconnectTimer) return;
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null;
        this.reconnectDelay = Math.min(this.reconnectDelay * 2, RECONNECT_MAX);
        this._open();
      }, this.reconnectDelay);
    }

    _startPing() {
      this._stopPing();
      this.pingTimer = setInterval(() => {
        if (this.ws && this.ws.readyState === 1) this.send({ op: 'ping' });
      }, PING_INTERVAL);
    }
    _stopPing() {
      if (this.pingTimer) { clearInterval(this.pingTimer); this.pingTimer = null; }
    }

    send(obj) {
      if (!this.ws || this.ws.readyState !== 1) return false;
      try {
        this.ws.send(JSON.stringify(obj));
        return true;
      } catch (e) {
        return false;
      }
    }

    /** Send now if open, else queue and flush on next open (fixes first-msg race). */
    sendWhenReady(obj) {
      if (this.send(obj)) return true;
      this.pending.push(obj);
      return false;
    }

    close() {
      this.chatId = null;
      if (this.reconnectTimer) { clearTimeout(this.reconnectTimer); this.reconnectTimer = null; }
      this._stopPing();
      if (this.ws) {
        try { this.ws.close(); } catch (_) {}
        this.ws = null;
      }
    }
  }

  window.ISAA = window.ISAA || {};
  window.ISAA.WS = new WS();
})();
