(function () {
  if (typeof marked === 'undefined' || typeof hljs === 'undefined') {
    console.warn('[md-render] marked or highlight.js not loaded - disabled');
    return;
  }

  marked.setOptions({
    gfm: true,
    breaks: true,
    highlight: function (code, lang) {
      if (lang && hljs.getLanguage(lang)) {
        return hljs.highlight(code, { language: lang }).value;
      }
      return hljs.highlightAuto(code).value;
    }
  });

  function renderElement(el) {
    if (el.dataset.mdParsed) return;
    var raw = el.textContent;
    var html = marked.parse(raw);
    el.innerHTML = html;
    el.dataset.mdParsed = 'true';
    el.querySelectorAll('pre code').forEach(function (block) {
      hljs.highlightElement(block);
    });
  }

  document.querySelectorAll('.step-body').forEach(renderElement);

  var observer = new MutationObserver(function (mutations) {
    for (var i = 0; i < mutations.length; i++) {
      var m = mutations[i];
      if (m.type === 'childList') {
        m.addedNodes.forEach(function (node) {
          if (node.nodeType === Node.ELEMENT_NODE) {
            if (node.matches('.step-body')) {
              renderElement(node);
            } else {
              node.querySelectorAll('.step-body').forEach(renderElement);
            }
          }
        });
      } else if (m.type === 'characterData') {
        var el = m.target.parentElement;
        if (el && el.matches('.step-body')) {
          renderElement(el);
        }
      }
    }
  });

  observer.observe(document.body, { childList: true, subtree: true });
})();
