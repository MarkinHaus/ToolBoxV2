// ToolBox Browser Extension - Smart Search Engine
// AI-powered webpage search with ISAA-2 integration

class TBSearchEngine {
    constructor() {
        this.searchHistory = [];
        this.currentResults = [];
        this.searchIndex = new Map();
        this.contextCache = new Map();
        this.settings = {
            maxResults: 10,
            highlightColor: '#ffeb3b',
            searchDelay: 300,
            enableAI: true,
            enableContextual: true,
            enableSemantic: true
        };

        this.init();
    }

    async init() {
        try {
            await this.loadSettings();
            this.buildPageIndex();
            this.setupEventListeners();

            TBUtils.info('SearchEngine', 'Smart search engine initialized');
        } catch (error) {
            TBUtils.handleError('SearchEngine', error);
        }
    }

    async loadSettings() {
        const stored = await TBUtils.getStorage([
            'search_max_results',
            'search_highlight_color',
            'search_delay',
            'search_enable_ai',
            'search_enable_contextual',
            'search_enable_semantic'
        ]);

        this.settings = {
            ...this.settings,
            maxResults: stored.search_max_results || this.settings.maxResults,
            highlightColor: stored.search_highlight_color || this.settings.highlightColor,
            searchDelay: stored.search_delay || this.settings.searchDelay,
            enableAI: stored.search_enable_ai !== false,
            enableContextual: stored.search_enable_contextual !== false,
            enableSemantic: stored.search_enable_semantic !== false
        };
    }

    setupEventListeners() {
        // Listen for page changes
        let lastUrl = location.href;
        new MutationObserver(() => {
            const url = location.href;
            if (url !== lastUrl) {
                lastUrl = url;
                this.buildPageIndex();
            }
        }).observe(document, { subtree: true, childList: true });

        // Debounced search function
        this.debouncedSearch = TBUtils.debounce(
            (query, options) => this.performSearch(query, options),
            this.settings.searchDelay
        );
    }

    buildPageIndex() {
        TBUtils.info('SearchEngine', 'Building page search index');

        this.searchIndex.clear();
        const elements = document.querySelectorAll('*');

        elements.forEach((element, index) => {
            const text = TBUtils.extractTextFromElement(element);
            if (text && text.length > 3) {
                this.searchIndex.set(index, {
                    element,
                    text: text.toLowerCase(),
                    tag: element.tagName.toLowerCase(),
                    classes: Array.from(element.classList),
                    id: element.id,
                    position: this.getElementPosition(element)
                });
            }
        });

        TBUtils.info('SearchEngine', `Indexed ${this.searchIndex.size} elements`);
    }

    getElementPosition(element) {
        const rect = element.getBoundingClientRect();
        return {
            top: rect.top + window.scrollY,
            left: rect.left + window.scrollX,
            width: rect.width,
            height: rect.height
        };
    }

    async search(query, options = {}) {
        if (!query || query.length < 2) {
            this.clearResults();
            return [];
        }

        TBUtils.info('SearchEngine', `Searching for: "${query}"`);

        // Add to search history
        this.addToHistory(query);

        // Perform search with debouncing
        this.debouncedSearch(query, options);

        return this.currentResults;
    }

    async performSearch(query, options = {}) {
        const startTime = performance.now();

        try {
            // Clear previous results
            this.clearHighlights();

            // Perform different types of searches
            const results = await Promise.all([
                this.textSearch(query, options),
                this.settings.enableContextual ? this.contextualSearch(query, options) : [],
                this.settings.enableSemantic ? this.semanticSearch(query, options) : [],
                this.settings.enableAI ? this.aiSearch(query, options) : []
            ]);

            // Merge and rank results
            this.currentResults = this.mergeAndRankResults(results.flat(), query);

            // Highlight results
            this.highlightResults(this.currentResults);

            // Update UI
            this.updateSearchUI(this.currentResults, query);

            const endTime = performance.now();
            TBUtils.info('SearchEngine', `Search completed in ${(endTime - startTime).toFixed(2)}ms`);

            return this.currentResults;

        } catch (error) {
            TBUtils.handleError('SearchEngine', error);
            return [];
        }
    }

    textSearch(query, options = {}) {
        const results = [];
        const queryLower = query.toLowerCase();
        const words = queryLower.split(/\s+/);

        for (const [index, data] of this.searchIndex) {
            let score = 0;
            let matches = 0;

            // Exact phrase match
            if (data.text.includes(queryLower)) {
                score += 100;
                matches++;
            }

            // Word matches
            words.forEach(word => {
                if (data.text.includes(word)) {
                    score += 10;
                    matches++;
                }
            });

            // Boost score based on element importance
            if (data.tag === 'h1') score *= 3;
            else if (data.tag === 'h2') score *= 2.5;
            else if (data.tag === 'h3') score *= 2;
            else if (data.tag === 'title') score *= 4;
            else if (data.tag === 'strong' || data.tag === 'b') score *= 1.5;

            if (score > 0) {
                results.push({
                    element: data.element,
                    score,
                    matches,
                    type: 'text',
                    snippet: this.generateSnippet(data.text, queryLower),
                    position: data.position
                });
            }
        }

        return results;
    }

    async contextualSearch(query, options = {}) {
        // Analyze page context and search based on current page content
        const pageContext = this.analyzePageContext();
        const contextualQuery = `${query} ${pageContext.keywords.join(' ')}`;

        return this.textSearch(contextualQuery, { ...options, contextual: true });
    }

    async semanticSearch(query, options = {}) {
        // Use ISAA for semantic understanding
        try {
            const response = await this.callISAA('semantic_search', {
                query,
                page_content: this.getPageContent(),
                context: this.analyzePageContext()
            });

            if (response && response.results) {
                return response.results.map(result => ({
                    ...result,
                    type: 'semantic',
                    element: this.findElementByContent(result.content)
                }));
            }
        } catch (error) {
            TBUtils.warn('SearchEngine', 'Semantic search failed', error);
        }

        return [];
    }

    async aiSearch(query, options = {}) {
        // Use ISAA-2 for intelligent search
        try {
            const response = await this.callISAA('intelligent_search', {
                query,
                page_url: window.location.href,
                page_title: document.title,
                page_content: this.getPageContent(),
                search_intent: this.analyzeSearchIntent(query)
            });

            if (response && response.suggestions) {
                return response.suggestions.map(suggestion => ({
                    element: this.findElementBySelector(suggestion.selector),
                    score: suggestion.confidence * 100,
                    type: 'ai',
                    snippet: suggestion.description,
                    action: suggestion.action
                }));
            }
        } catch (error) {
            TBUtils.warn('SearchEngine', 'AI search failed', error);
        }

        return [];
    }

    mergeAndRankResults(results, query) {
        // Remove duplicates and sort by score
        const uniqueResults = new Map();

        results.forEach(result => {
            if (result.element) {
                const key = result.element;
                const existing = uniqueResults.get(key);

                if (!existing || result.score > existing.score) {
                    uniqueResults.set(key, result);
                }
            }
        });

        return Array.from(uniqueResults.values())
            .sort((a, b) => b.score - a.score)
            .slice(0, this.settings.maxResults);
    }

    highlightResults(results) {
        results.forEach((result, index) => {
            if (result.element && result.element.nodeType === Node.ELEMENT_NODE) {
                result.element.classList.add('tb-search-result');
                result.element.setAttribute('data-tb-result-index', index);
                result.element.style.backgroundColor = this.settings.highlightColor;
                result.element.style.transition = 'background-color 0.3s ease';
            }
        });
    }

    clearHighlights() {
        document.querySelectorAll('.tb-search-result').forEach(element => {
            element.classList.remove('tb-search-result');
            element.removeAttribute('data-tb-result-index');
            element.style.backgroundColor = '';
        });
    }

    clearResults() {
        this.clearHighlights();
        this.currentResults = [];
        this.updateSearchUI([], '');
    }

    generateSnippet(text, query, maxLength = 150) {
        const index = text.indexOf(query);
        if (index === -1) return text.substring(0, maxLength) + '...';

        const start = Math.max(0, index - 50);
        const end = Math.min(text.length, index + query.length + 50);

        let snippet = text.substring(start, end);
        if (start > 0) snippet = '...' + snippet;
        if (end < text.length) snippet = snippet + '...';

        return snippet;
    }

    analyzePageContext() {
        const title = document.title;
        const headings = Array.from(document.querySelectorAll('h1, h2, h3'))
            .map(h => h.textContent.trim());
        const meta = Array.from(document.querySelectorAll('meta[name="keywords"], meta[name="description"]'))
            .map(m => m.content);

        return {
            title,
            headings,
            keywords: [...headings, ...meta].join(' ').split(/\s+/).slice(0, 20),
            domain: window.location.hostname,
            path: window.location.pathname
        };
    }

    analyzeSearchIntent(query) {
        const intents = {
            navigation: /^(go to|navigate to|find|show me)/i,
            information: /^(what|how|why|when|where|who)/i,
            action: /^(click|press|submit|download|buy|purchase)/i,
            search: /^(search|look for|find)/i
        };

        for (const [intent, pattern] of Object.entries(intents)) {
            if (pattern.test(query)) {
                return intent;
            }
        }

        return 'general';
    }

    getPageContent() {
        return {
            title: document.title,
            url: window.location.href,
            text: document.body.innerText.substring(0, 5000), // Limit content size
            headings: Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'))
                .map(h => ({ tag: h.tagName, text: h.textContent.trim() })),
            links: Array.from(document.querySelectorAll('a[href]'))
                .map(a => ({ text: a.textContent.trim(), href: a.href })).slice(0, 50)
        };
    }

    findElementByContent(content) {
        for (const [index, data] of this.searchIndex) {
            if (data.text.includes(content.toLowerCase())) {
                return data.element;
            }
        }
        return null;
    }

    findElementBySelector(selector) {
        try {
            return document.querySelector(selector);
        } catch (error) {
            return null;
        }
    }

    async callISAA(function_name, data) {
        if (typeof chrome !== 'undefined' && chrome.runtime) {
            return await chrome.runtime.sendMessage({
                type: 'TB_ISAA_REQUEST',
                function: function_name,
                data,
                timestamp: Date.now()
            });
        }
        return null;
    }

    updateSearchUI(results, query) {
        // This will be handled by the UI manager
        if (window.tbUIManager) {
            window.tbUIManager.updateSearchResults(results, query);
        }
    }

    addToHistory(query) {
        this.searchHistory.unshift({
            query,
            timestamp: Date.now(),
            url: window.location.href,
            results: this.currentResults.length
        });

        // Keep only last 50 searches
        if (this.searchHistory.length > 50) {
            this.searchHistory = this.searchHistory.slice(0, 50);
        }
    }

    // Public API
    async smartSearch(query, options = {}) {
        return await this.search(query, { ...options, enableAI: true });
    }

    scrollToResult(index) {
        if (this.currentResults[index] && this.currentResults[index].element) {
            this.currentResults[index].element.scrollIntoView({
                behavior: 'smooth',
                block: 'center'
            });
        }
    }

    getSearchHistory() {
        return this.searchHistory;
    }

    async updateSettings(newSettings) {
        this.settings = { ...this.settings, ...newSettings };
        await TBUtils.setStorage({
            search_max_results: this.settings.maxResults,
            search_highlight_color: this.settings.highlightColor,
            search_delay: this.settings.searchDelay,
            search_enable_ai: this.settings.enableAI,
            search_enable_contextual: this.settings.enableContextual,
            search_enable_semantic: this.settings.enableSemantic
        });
    }
}

// Export for global use
if (typeof window !== 'undefined') {
    window.TBSearchEngine = TBSearchEngine;
}
