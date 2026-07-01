/* Kurse — mini framework. One file, no deps.
 *
 * Two modes, auto-detected:
 *   SHELL  (top window, learn.html): name gate, loads cohort sessions into an
 *          iframe, renders prev/next nav over the released sheets, forwards
 *          learner events to the backend.
 *   EMBED  (inside the session iframe): the API an authored task page uses —
 *          Kurse.task(i), Kurse.hint(task, level), Kurse.tasks(el, spec),
 *          plus auto-wiring of [data-kurse-task] / [data-kurse-hint].
 *
 * Timing is server-side: every task()/hint() call is one event; the server
 * closes the previous open event, so "hint open time" is real, not a peek.
 */
(function () {
  "use strict";
  var EMBED = window.top !== window.self;

  // ---- tiny fetch helpers --------------------------------------------------
  function post(url, body) {
    return fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" },
      credentials: "same-origin", body: JSON.stringify(body || {})
    }).then(function (r) { return r.json(); });
  }
  function getj(url) {
    return fetch(url, { credentials: "same-origin" }).then(function (r) { return r.json(); });
  }

  // =========================================================================
  // EMBED MODE — the authoring API
  // =========================================================================
  if (EMBED) {
    function send(m) { m.__kurse = 1; parent.postMessage(m, "*"); }

    var Kurse = {
      task: function (i) { send({ type: "task", taskIdx: i | 0 }); },
      hint: function (taskIdx, level) { send({ type: "hint", taskIdx: taskIdx | 0, hint: level | 0 }); },
      done: function () { send({ type: "done" }); },
      next: function () { send({ type: "nav", dir: 1 }); },
      prev: function () { send({ type: "nav", dir: -1 }); },

      /* Builder: turn a spec into a wired task list.
       * spec = [{ title, body(html), hints:[h1,h2,h3,h4], bonus?, extra? }, ...] */
      tasks: function (el, spec) {
        el.innerHTML = "";
        spec.forEach(function (t, i) {
          var card = document.createElement("section");
          card.setAttribute("data-kurse-task", i);
          card.className = "kurse-task";
          var hints = (t.hints || []).map(function (h, hi) {
            return '<details class="kurse-hint" data-kurse-hint="' + (hi + 1) +
              '"><summary>Tipp ' + (hi + 1) + (hi === 3 ? " · Lösung" : "") +
              '</summary><div>' + h + "</div></details>";
          }).join("");
          card.innerHTML =
            '<label class="kurse-check"><input type="checkbox"> <b>' +
            (i + 1) + ". " + (t.title || "") + "</b></label>" +
            '<div class="kurse-body">' + (t.body || "") + "</div>" +
            (t.bonus ? '<p class="kurse-bonus">⭐ ' + t.bonus + "</p>" : "") +
            (t.extra ? '<p class="kurse-extra">🌟 ' + t.extra + "</p>" : "") +
            (hints ? '<div class="kurse-hints">' + hints + "</div>" : "");
          el.appendChild(card);
        });
        wire(el);
        checkboxWatch(el);
      }
    };

    // auto-wire declarative pages: report on hint open + task reach
    function wire(root) {
      root = root || document;
      root.querySelectorAll("[data-kurse-hint]").forEach(function (h) {
        if (h.__kw) return; h.__kw = 1;
        var lvl = parseInt(h.getAttribute("data-kurse-hint"), 10) || 1;
        var task = closestTask(h);
        var fire = function () { Kurse.hint(task, lvl); };
        if (h.tagName === "DETAILS")
          h.addEventListener("toggle", function () { if (h.open) fire(); });
        else h.addEventListener("click", fire, { once: true });
      });
      root.querySelectorAll("[data-kurse-task]").forEach(function (t) {
        if (t.__kw) return; t.__kw = 1;
        var idx = parseInt(t.getAttribute("data-kurse-task"), 10) || 0;
        new IntersectionObserver(function (es, o) {
          es.forEach(function (e) { if (e.isIntersecting) { Kurse.task(idx); o.disconnect(); } });
        }, { threshold: 0.4 }).observe(t);
      });
    }
    function closestTask(el) {
      var p = el.closest("[data-kurse-task]");
      return p ? (parseInt(p.getAttribute("data-kurse-task"), 10) || 0) : 0;
    }
    function checkboxWatch(root) {
      var boxes = root.querySelectorAll('.kurse-check input[type=checkbox]');
      boxes.forEach(function (b) {
        b.addEventListener("change", function () {
          if (Array.prototype.every.call(boxes, function (x) { return x.checked; }))
            Kurse.done();
        });
      });
    }

    // Zero-edit compat: hook the workshop-page naming model.
    var doneSent = false;
    function done1() { if (!doneSent) { doneSent = true; Kurse.done(); } }

    function hookLegacy() {
      // hint reveal: reveal(btn, taskNum, hintNum)
      if (typeof window.reveal === "function" && !window.reveal.__kh) {
        var orig = window.reveal;
        var wrap = function (btn, t, h) {
          try { Kurse.hint(t, h); } catch (e) {}
          return orig.apply(this, arguments);
        };
        wrap.__kh = 1; window.reveal = wrap;
      }
      // completion: markDone(n) → celebration
      if (typeof window.markDone === "function" && !window.markDone.__kh) {
        var od = window.markDone;
        var wd = function () { var r = od.apply(this, arguments); done1(); return r; };
        wd.__kh = 1; window.markDone = wd;
      }
      // task sections id="task-N" → report when reached
      document.querySelectorAll('[id^="task-"]').forEach(function (sec) {
        if (sec.__kh) return; sec.__kh = 1;
        var idx = parseInt(sec.id.split("-")[1], 10);
        if (isNaN(idx)) return;
        new IntersectionObserver(function (es, o) {
          es.forEach(function (e) { if (e.isIntersecting) { Kurse.task(idx); o.disconnect(); } });
        }, { threshold: 0.35 }).observe(sec);
      });
      // fallback completion: all checkboxes ticked
      var boxes = document.querySelectorAll('input[type=checkbox]');
      boxes.forEach(function (b) {
        if (b.__kh) return; b.__kh = 1;
        b.addEventListener("change", function () {
          if (Array.prototype.every.call(boxes, function (x) { return x.checked; })) done1();
        });
      });
    }

    // In a srcdoc iframe, clicking <a href="#task-N"> navigates to
    // about:srcdoc#... and blanks the page. Intercept and scroll manually.
    document.addEventListener("click", function (e) {
      var a = e.target.closest && e.target.closest('a[href^="#"]');
      if (!a) return;
      var id = a.getAttribute("href").slice(1);
      if (!id) return;
      var el = document.getElementById(id);
      if (el) { e.preventDefault(); el.scrollIntoView({ behavior: "smooth", block: "start" }); }
    }, true);

    // shell can ask us to resume at a specific task after (re)load
    window.addEventListener("message", function (e) {
      var m = e.data;
      if (!m || typeof m.__kurse_goto !== "number") return;
      var el = document.getElementById("task-" + m.__kurse_goto);
      if (el) { el.scrollIntoView({ behavior: "instant", block: "start" }); Kurse.task(m.__kurse_goto); }
    });

    window.Kurse = Kurse;
    document.addEventListener("DOMContentLoaded", function () { wire(document); hookLegacy(); });
    if (document.readyState !== "loading") { wire(document); hookLegacy(); }
    return;
  }

  // =========================================================================
  // SHELL MODE — learn.html
  // =========================================================================
  var st = { coid: null, name: null, cid: null, sessions: [], idx: 0 };

  function coidFromPath() {
    var m = location.pathname.match(/\/l\/([^\/?#]+)/);
    return m ? m[1] : null;
  }
  var lsKey = function () { return "kurse:name:" + st.coid; };

  function boot(root) {
    st.coid = coidFromPath();
    st.root = root;
    var saved = localStorage.getItem(lsKey());
    if (saved) join(saved); else nameGate();
  }

  function nameGate() {
    st.root.innerHTML =
      '<div class="k-gate"><h1>Willkommen 👋</h1>' +
      "<p>Gib deinen Namen ein, um zu starten:</p>" +
      '<input id="k-name" placeholder="Dein Name" autofocus>' +
      '<button id="k-go">Los geht\'s</button></div>';
    var inp = st.root.querySelector("#k-name");
    var go = function () { if (inp.value.trim()) join(inp.value.trim()); };
    st.root.querySelector("#k-go").onclick = go;
    inp.addEventListener("keydown", function (e) { if (e.key === "Enter") go(); });
  }

  function join(name) {
    post("/api/join", { coid: st.coid, name: name }).then(function (r) {
      if (r.error) { localStorage.removeItem(lsKey()); return nameGate(); }
      st.name = name; st.cid = r.cid; st.sessions = r.sessions;
      st.resume = r.resume || null;
      localStorage.setItem(lsKey(), name);
      shell();
      go(r.resume ? r.resume.pos : r.anchor);
    });
  }

  function shell() {
    st.root.innerHTML =
      '<header class="k-bar">' +
      '<button id="k-prev">◀</button>' +
      '<div id="k-dots" class="k-dots"></div>' +
      '<button id="k-next">▶</button>' +
      '<span class="k-me"></span></header>' +
      '<iframe id="k-stage" title="Aufgabe"></iframe>' +
      '<div id="k-cheer" class="k-cheer" hidden>🎉 Stark! Alle Aufgaben erledigt.</div>';
    st.root.querySelector(".k-me").textContent = st.name;
    st.root.querySelector("#k-prev").onclick = function () { go(st.idx - 1); };
    st.root.querySelector("#k-next").onclick = function () { go(st.idx + 1); };
    renderDots();
    window.addEventListener("message", onEmbed);
  }

  function renderDots() {
    var d = st.root.querySelector("#k-dots");
    d.innerHTML = "";
    st.sessions.forEach(function (s, i) {
      var b = document.createElement("button");
      b.className = "k-dot" + (i === st.idx ? " on" : "");
      b.textContent = i + 1;
      b.title = s.name;
      b.onclick = function () { go(i); };
      d.appendChild(b);
    });
    st.root.querySelector("#k-prev").disabled = st.idx <= 0;
    st.root.querySelector("#k-next").disabled = st.idx >= st.sessions.length - 1;
  }

  function go(idx) {
    if (idx < 0 || idx >= st.sessions.length) return;
    st.idx = idx;
    st.root.querySelector("#k-cheer").hidden = true;
    getj("/api/session/" + st.coid + "/" + idx + "?name=" + encodeURIComponent(st.name))
      .then(function (r) {
        if (r.error) return;
        var frame = st.root.querySelector("#k-stage");
        var resumeTask = (st.resume && idx === st.resume.pos && st.resume.task > 0)
          ? st.resume.task : 0;
        st.resume = null;                       // one-shot: only on initial resume
        frame.onload = function () {
          if (resumeTask > 0 && frame.contentWindow)
            frame.contentWindow.postMessage({ __kurse_goto: resumeTask }, "*");
        };
        frame.srcdoc =
          '<!doctype html><meta charset="utf-8">' +
          '<script src="/static/kurse.js"><\/script>' + r.html;
        renderDots();
      });
  }

  function onEmbed(e) {
    var m = e.data;
    if (!m || !m.__kurse) return;
    if (m.type === "nav") return go(st.idx + (m.dir || 0));
    if (m.type === "done") { st.root.querySelector("#k-cheer").hidden = false; return; }
    // task / hint → backend event log (server timestamps + timing)
    post("/api/progress", {
      coid: st.coid, name: st.name,
      event: { type: m.type, sIdx: st.idx, taskIdx: m.taskIdx || 0, hint: m.hint || 0 }
    });
  }

  window.Kurse = { boot: boot };
})();
