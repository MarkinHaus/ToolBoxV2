import {marked} from '/web/node_modules/marked/src/marked.js';
import {markedHighlight} from '/web/node_modules/marked-highlight/src/index.js';


marked.use(markedHighlight({
    langPrefix: 'hljs language-',
    highlight(code, lang) {
        const language = hljs.getLanguage(lang) ? lang : 'plaintext';
        return hljs.highlight(code, { language }).value;
    }
}));

function renderMarkdown(markdownString) {
    return marked(markdownString);
}

const markdownVisibleEvent = new CustomEvent('markdownVisible');

function handleMarkdownVisible(event) {
    const element = event.target;
    element.innerHTML = renderMarkdown(element.innerHTML);
}

function observeMarkdownElements(observer) {
    const autoMarkdownElements = document.querySelectorAll('.autoMarkdownVisible');

    autoMarkdownElements.forEach(autoMarkdownElement => {
        autoMarkdownElement.querySelectorAll('.markdown').forEach(element => {
            if (!element.observed) {
                element.addEventListener('markdownVisible', handleMarkdownVisible);
                observer.observe(element);
                element.observed = true;
            }
        });
    });
}

export function setupMutationObserver() {
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1,
    };

    const observerCallback = (entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.dispatchEvent(markdownVisibleEvent);
                observer.unobserve(entry.target);
            }
        });
    };

    const intersectionObserver = new IntersectionObserver(observerCallback, observerOptions);
    observeMarkdownElements(intersectionObserver);

    const mutationObserverCallback = (mutationsList, observer) => {
        for (const mutation of mutationsList) {
            if (mutation.type === 'childList') {
                observeMarkdownElements(intersectionObserver);
            }
        }
    };

    const mutationObserver = new MutationObserver(mutationObserverCallback);
    mutationObserver.observe(document.body, { childList: true, subtree: true });
}

document.addEventListener('DOMContentLoaded', ()=>{
    const initMDBtn = document.getElementById('initMDBtn');
    if (initMDBtn){
        initMDBtn.addEventListener('click', ()=> {
            setupMutationObserver();
            initMDBtn.remove()
        });
    }else {
        const initMDBtn = undefined;
        console.log("Starting AutoMarkdown")
        setTimeout(()=>{
            setupMutationObserver()
            console.log("AutoMarkdown: [started]")
        }, 2)//4500)
    }
})

