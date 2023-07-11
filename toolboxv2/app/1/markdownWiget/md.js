import {marked} from '/app/node_modules/marked/src/marked.js';
import markedHighlight from '/app/node_modules/marked-highlight/src/index.js';
import {hljs} from "highlight.js";


marked.use(markedHighlight({
    langPrefix: 'hljs language-',
    highlight(code, lang) {
        const language = hljs.getLanguage(lang) ? lang : 'plaintext';
        return hljs.highlight(code, { language }).value;
    }
}));


const liveMarkdownInput = document.getElementById('live-markdown-input');
const markdownOutput = document.getElementById('markdown-output');

liveMarkdownInput.addEventListener('input', () => {
    markdownOutput.innerHTML = marked(liveMarkdownInput.value, {headerIds: false, mangle: false});
});

function renderMarkdown(markdown) {
    const outputDiv = document.getElementById('markdown-output');
    outputDiv.innerHTML = marked(markdown);
}
