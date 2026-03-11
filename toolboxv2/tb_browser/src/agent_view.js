// agent_view.js — TB Web Agent: Structured DOM View + Delta Capture
// Uses chrome.debugger CDP for AX Tree access

'use strict';

class AgentView {
  constructor(tabId) {
    this.tabId = tabId;
    this.attached = false;
    this._lastSnapshot = null;
  }

  // Attach CDP debugger
  async attach() {
    if (this.attached) return;
    await chrome.debugger.attach({ tabId: this.tabId }, '1.3');
    this.attached = true;
  }

  async detach() {
    if (!this.attached) return;
    await chrome.debugger.detach({ tabId: this.tabId });
    this.attached = false;
  }

  async _send(method, params = {}) {
    return chrome.debugger.sendCommand({ tabId: this.tabId }, method, params);
  }

  // --- STRUCTURED VIEW (Initial Page Snapshot) ---

  async getStructuredView(maxDepth = 5) {
    await this.attach();
    try {
      const { nodes } = await this._send('Accessibility.getFullAXTree');
      const tree = this._buildTree(nodes, maxDepth);
      this._lastSnapshot = { nodes, tree, ts: Date.now() };
      return {
        compressed: this._renderCompressed(tree),
        interactable: this._extractInteractable(nodes),
        metadata: await this._getPageMeta()
      };
    } catch (e) {
      // Fallback to DOM-based snapshot
      return this._getDOMFallback(maxDepth);
    }
  }

  _buildTree(nodes, maxDepth) {
    const byId = {};
    for (const n of nodes) byId[n.nodeId] = n;
    // Find roots
    const childIds = new Set(nodes.flatMap(n => n.childIds || []));
    const roots = nodes.filter(n => !childIds.has(n.nodeId));
    return roots.map(r => this._buildNode(r, byId, 0, maxDepth));
  }

  _buildNode(node, byId, depth, maxDepth) {
    if (depth > maxDepth) return null;
    // Skip hidden/irrelevant nodes
    if (this._isHidden(node)) return null;

    const result = {
      role: node.role?.value,
      name: this._getName(node),
      nodeId: node.nodeId,
      properties: this._getKeyProps(node),
      children: []
    };

    const children = (node.childIds || [])
      .map(id => byId[id])
      .filter(Boolean)
      .map(c => this._buildNode(c, byId, depth + 1, maxDepth))
      .filter(Boolean);

    // Collapse single-child wrappers with no meaningful role
    if (children.length === 1 && !result.role && !result.name) {
      return children[0];
    }
    result.children = children;
    return result;
  }

  _isHidden(node) {
    const hidden = node.properties?.find(p => p.name === 'hidden');
    if (hidden?.value?.value === true) return true;
    const role = node.role?.value;
    return role === 'none' || role === 'presentation';
  }

  _getName(node) {
    const name = node.name?.value;
    if (!name) return null;
    return name.length > 80 ? name.slice(0, 80) + '…' : name;
  }

  _getKeyProps(node) {
    const interestingProps = ['disabled', 'checked', 'selected', 'expanded',
                              'required', 'invalid', 'focused', 'value'];
    const props = {};
    for (const p of (node.properties || [])) {
      if (interestingProps.includes(p.name) && p.value?.value !== undefined) {
        props[p.name] = p.value.value;
      }
    }
    return Object.keys(props).length ? props : null;
  }

  _renderCompressed(tree, indent = 0) {
    if (!tree) return '';
    const pad = '  '.repeat(indent);
    const role = tree.role || 'div';
    const name = tree.name ? ` "${tree.name}"` : '';
    const props = tree.properties
      ? ' [' + Object.entries(tree.properties).map(([k,v]) => `${k}=${v}`).join(', ') + ']'
      : '';

    let line = `${pad}${role}${name}${props}`;
    if (!tree.children?.length) return line;

    // Collapse lists of >5 identical roles
    const childLines = this._collapseRepeated(tree.children)
      .map(c => this._renderCompressed(c, indent + 1))
      .filter(Boolean);

    return line + '\n' + childLines.join('\n');
  }

  _collapseRepeated(children) {
    if (children.length <= 4) return children;
    const first = children[0];
    const allSame = children.every(c => c.role === first.role && !c.name);
    if (allSame) return [...children.slice(0, 2), { role: `[...${children.length - 2} more ${first.role}]`, name: null, children: [] }];
    return children;
  }

  _extractInteractable(nodes) {
    const interactableRoles = ['button', 'link', 'textbox', 'combobox', 'checkbox',
                               'radio', 'menuitem', 'tab', 'searchbox', 'switch'];
    return nodes
      .filter(n => interactableRoles.includes(n.role?.value))
      .filter(n => !this._isHidden(n))
      .map(n => ({
        nodeId: n.nodeId,
        role: n.role?.value,
        name: this._getName(n),
        props: this._getKeyProps(n)
      }));
  }

  async _getPageMeta() {
    const { result } = await this._send('Runtime.evaluate', {
      expression: 'JSON.stringify({title: document.title, url: location.href, readyState: document.readyState})'
    });
    try { return JSON.parse(result.value); } catch { return {}; }
  }

  async _getDOMFallback(maxDepth) {
    const { result } = await this._send('Runtime.evaluate', {
      expression: `(function(depth) {
        function visit(el, d) {
          if (d > depth || el.hidden || el.offsetParent === null) return null;
          const tag = el.tagName.toLowerCase();
          const text = (el.innerText || '').trim().slice(0, 60);
          const attrs = {};
          if (el.role) attrs.role = el.role;
          if (el.disabled) attrs.disabled = true;
          if (el.href) attrs.href = el.href.slice(0, 50);
          const children = [...el.children].map(c => visit(c, d+1)).filter(Boolean);
          if (!text && !children.length && !Object.keys(attrs).length) return null;
          return { tag, text: text || null, attrs, children };
        }
        return JSON.stringify(visit(document.body, 0));
      })(${maxDepth})`
    });
    return { compressed: result.value, interactable: [], metadata: {} };
  }

  // --- DELTA VIEW (After Action) ---

  async captureActionDelta(actionFn) {
    await this.attach();
    const before = await this._snapshotNodes();
    const beforeFocus = await this._getFocus();

    await actionFn();
    await this._waitForSettle();

    const after = await this._snapshotNodes();
    const afterFocus = await this._getFocus();

    const delta = this._diff(before, after);
    this._lastSnapshot = { nodes: after, ts: Date.now() };

    return {
      changed: delta.changed,
      added: delta.added,
      removed: delta.removed,
      focus_changed: beforeFocus !== afterFocus,
      new_focus: afterFocus,
      summary: this._renderDelta(delta, beforeFocus, afterFocus)
    };
  }

  async _snapshotNodes() {
    try {
      const { nodes } = await this._send('Accessibility.getFullAXTree');
      const map = {};
      for (const n of nodes) map[n.nodeId] = n;
      return map;
    } catch { return {}; }
  }

  async _getFocus() {
    try {
      const { result } = await this._send('Runtime.evaluate', {
        expression: `document.activeElement ? document.activeElement.tagName + ':' + (document.activeElement.innerText || document.activeElement.value || '').slice(0, 40) : 'none'`
      });
      return result.value;
    } catch { return 'unknown'; }
  }

  _diff(before, after) {
    const added = [];
    const removed = [];
    const changed = [];

    const beforeIds = new Set(Object.keys(before));
    const afterIds = new Set(Object.keys(after));

    for (const id of afterIds) {
      if (!beforeIds.has(id)) {
        const n = after[id];
        if (!this._isHidden(n)) added.push(this._summarizeNode(n));
      } else {
        const b = before[id], a = after[id];
        if (this._nodeChanged(b, a)) {
          changed.push({
            nodeId: id,
            role: a.role?.value,
            name: this._getName(a),
            diff: this._propDiff(b, a)
          });
        }
      }
    }

    for (const id of beforeIds) {
      if (!afterIds.has(id)) {
        const n = before[id];
        if (!this._isHidden(n)) removed.push(this._summarizeNode(n));
      }
    }

    return { added, removed, changed };
  }

  _nodeChanged(before, after) {
    return JSON.stringify(before.properties) !== JSON.stringify(after.properties)
      || before.name?.value !== after.name?.value;
  }

  _propDiff(before, after) {
    const bProps = this._getKeyProps(before) || {};
    const aProps = this._getKeyProps(after) || {};
    const diff = {};
    const allKeys = new Set([...Object.keys(bProps), ...Object.keys(aProps)]);
    for (const k of allKeys) {
      if (bProps[k] !== aProps[k]) diff[k] = `${bProps[k]} → ${aProps[k]}`;
    }
    if (before.name?.value !== after.name?.value) {
      diff.name = `"${before.name?.value}" → "${after.name?.value}"`;
    }
    return diff;
  }

  _summarizeNode(node) {
    return {
      nodeId: node.nodeId,
      role: node.role?.value,
      name: this._getName(node)
    };
  }

  _renderDelta(delta, beforeFocus, afterFocus) {
    const lines = [];
    for (const n of delta.added) {
      lines.push(`+ ${n.role} "${n.name || ''}" [new]`);
    }
    for (const n of delta.removed) {
      lines.push(`- ${n.role} "${n.name || ''}" [removed]`);
    }
    for (const n of delta.changed) {
      const diffs = Object.entries(n.diff).map(([k,v]) => `${k}: ${v}`).join(', ');
      lines.push(`~ ${n.role} "${n.name || ''}": ${diffs}`);
    }
    if (beforeFocus !== afterFocus) {
      lines.push(`FOCUS: ${afterFocus}`);
    }
    return lines.join('\n') || 'no visible changes';
  }

  // Wait for DOM to settle after an action
  _waitForSettle(minWait = 200, maxWait = 3000) {
    return new Promise(resolve => {
      const start = Date.now();
      // Inject observer script
      this._send('Runtime.evaluate', {
        expression: `new Promise(res => {
          let t;
          const obs = new MutationObserver(() => {
            clearTimeout(t);
            t = setTimeout(() => { obs.disconnect(); res('settled'); }, 100);
          });
          obs.observe(document.body, { subtree: true, childList: true, attributes: true, characterData: true });
          setTimeout(() => { obs.disconnect(); res('timeout'); }, ${maxWait});
        })`,
        awaitPromise: true
      }).then(() => {
        const elapsed = Date.now() - start;
        if (elapsed < minWait) {
          setTimeout(resolve, minWait - elapsed);
        } else {
          resolve();
        }
      }).catch(resolve);
    });
  }

  // Execute a structured action
  async executeAction(action) {
    await this.attach();
    switch (action.type) {
      case 'click':
        return this.captureActionDelta(() => this._clickNode(action.target));
      case 'type':
        return this.captureActionDelta(() => this._typeInNode(action.target, action.value));
      case 'navigate':
        return this.captureActionDelta(() => this._navigate(action.value));
      case 'scroll':
        return this.captureActionDelta(() => this._scroll(action.direction));
      case 'extract':
        return this._extractContent(action.target);
      default:
        throw new Error(`Unknown action type: ${action.type}`);
    }
  }

  async _clickNode(target) {
    // Get coordinates via DOM node
    const { result } = await this._send('Runtime.evaluate', {
      expression: `(function() {
        // Find by aria label or text content
        const all = document.querySelectorAll('button, a, [role="button"], [role="link"], input, select');
        for (const el of all) {
          const text = (el.innerText || el.value || el.getAttribute('aria-label') || '').trim();
          if (text === ${JSON.stringify(target.name || '')}) {
            const r = el.getBoundingClientRect();
            el.click();
            return JSON.stringify({x: r.x + r.width/2, y: r.y + r.height/2, clicked: true});
          }
        }
        return JSON.stringify({clicked: false});
      })()`
    });
    return JSON.parse(result.value || '{}');
  }

  async _typeInNode(target, value) {
    await this._send('Runtime.evaluate', {
      expression: `(function() {
        const el = document.querySelector('input, textarea, [contenteditable]');
        if (el) { el.focus(); el.value = ${JSON.stringify(value)}; el.dispatchEvent(new InputEvent('input', {bubbles: true})); }
      })()`
    });
  }

  async _navigate(url) {
    await this._send('Runtime.evaluate', {
      expression: `location.href = ${JSON.stringify(url)}`
    });
  }

  async _scroll(direction) {
    const amount = direction === 'down' ? 400 : -400;
    await this._send('Runtime.evaluate', {
      expression: `window.scrollBy(0, ${amount})`
    });
  }

  async _extractContent(target) {
    const { result } = await this._send('Runtime.evaluate', {
      expression: `(function() {
        const sel = ${JSON.stringify(target?.selector || 'body')};
        const el = document.querySelector(sel);
        return el ? el.innerText.slice(0, 2000) : null;
      })()`
    });
    return { content: result.value };
  }
}

export default AgentView
