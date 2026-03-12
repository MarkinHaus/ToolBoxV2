/**
 * popup_context_chat.unit.test.js
 *
 * Unit-Tests für:
 *   1. getFreshPageContext — null-Handling, Fehlerbehandlung
 *   2. sendChatMessage    — context → mini_task/user_task Mapping
 *   3. executeStructuredAction (Agent Mode) — format_schema Payload
 *
 * Ausführen: npm run test:unit
 */

// ─── Chrome Mock ─────────────────────────────────────────────────────────────

function buildChromeMock({
    tabsResponse = { success: true, url: 'https://example.com', title: 'Example', summary: {} },
    tabsError    = null,
    runtimeResponse = null,
    runtimeError    = null,
    activeTab    = { id: 1, url: 'https://example.com', title: 'Example' },
} = {}) {
    return {
        tabs: {
            query: jest.fn().mockImplementation((_, cb) => cb([activeTab])),
            sendMessage: jest.fn().mockImplementation((tabId, msg, cb) => {
                if (tabsError) {
                    global.chrome.runtime.lastError = { message: tabsError };
                    cb(undefined);
                    global.chrome.runtime.lastError = null;
                } else {
                    cb(tabsResponse);
                }
            }),
        },
        runtime: {
            lastError: null,
            sendMessage: jest.fn().mockImplementation((msg, cb) => {
                if (runtimeError) {
                    global.chrome.runtime.lastError = { message: runtimeError };
                    cb(undefined);
                    global.chrome.runtime.lastError = null;
                } else {
                    cb(runtimeResponse ?? { success: true, data: { result: { data: 'PONG' } } });
                }
            }),
        },
        storage: {
            sync:  { get: jest.fn().mockResolvedValue({}), set: jest.fn() },
            local: { get: jest.fn().mockResolvedValue({}), set: jest.fn() },
        },
    };
}

// ─── Isolierter Extrakt der relevanten Popup-Logik ───────────────────────────
// Kein Import des echten popup.js (DOM + chrome APIs fehlen im Test-Env).
// Wir testen die Kern-Transformationen direkt.

function buildContextSummary(freshContext) {
    if (!freshContext) return 'Kein Seitenkontext verfügbar.';
    return `URL: ${freshContext.url || ''}, Title: ${freshContext.title || ''}, Summary: ${JSON.stringify(freshContext.summary || {})}`;
}

function buildChatPayload({ freshContext, message, voiceLanguage, agentName, chatHistory }) {
    const contextSummary = buildContextSummary(freshContext);
    return {
        mini_task:       `Du bist ein Browser-Assistent. Antworte auf Deutsch (${voiceLanguage}). Seitenkontext: ${contextSummary}`,
        user_task:       message,
        agent_name:      agentName || 'speed',
        task_from:       'browser_extension',
        message_history: chatHistory || [],
    };
}

async function getFreshPageContext(chromeMock) {
    return new Promise((resolve) => {
        chromeMock.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            if (!tabs || !tabs[0]) { resolve(null); return; }
            chromeMock.tabs.sendMessage(tabs[0].id, { type: 'GET_PAGE_CONTEXT' }, (response) => {
                if (chromeMock.runtime.lastError || !response || !response.success) {
                    resolve(null);
                } else {
                    resolve(response);
                }
            });
        });
    });
}

async function makeAPICall(chromeMock, endpoint, method, data) {
    return new Promise((resolve, reject) => {
        chromeMock.runtime.sendMessage(
            { type: 'API_REQUEST', data: { endpoint, method, body: data } },
            (response) => {
                if (chromeMock.runtime.lastError) {
                    reject(new Error(chromeMock.runtime.lastError.message));
                    return;
                }
                if (!response || !response.success) {
                    if (response?.data) { resolve(response.data); return; }
                    reject(new Error(response?.error || 'API call failed'));
                    return;
                }
                resolve(response.data);
            }
        );
    });
}

// ─── 1. getFreshPageContext ───────────────────────────────────────────────────

describe('getFreshPageContext', () => {
    test('gibt null zurück wenn kein aktiver Tab vorhanden', async () => {
        const mock = buildChromeMock({ activeTab: null });
        mock.tabs.query = jest.fn().mockImplementation((_, cb) => cb([]));
        const ctx = await getFreshPageContext(mock);
        expect(ctx).toBeNull();
    });

    test('gibt null zurück bei Connection-Fehler (chrome://-Seite)', async () => {
        const mock = buildChromeMock({ tabsError: 'Could not establish connection. Receiving end does not exist.' });
        const ctx = await getFreshPageContext(mock);
        expect(ctx).toBeNull();
    });

    test('gibt null zurück wenn response.success = false', async () => {
        const mock = buildChromeMock({ tabsResponse: { success: false } });
        const ctx = await getFreshPageContext(mock);
        expect(ctx).toBeNull();
    });

    test('gibt Context-Objekt zurück bei Erfolg', async () => {
        const expected = {
            success: true,
            url: 'https://example.com',
            title: 'Example Page',
            summary: { headings: ['Hello'], elementCount: 42 },
        };
        const mock = buildChromeMock({ tabsResponse: expected });
        const ctx = await getFreshPageContext(mock);
        expect(ctx).toMatchObject({ success: true, url: 'https://example.com' });
    });

    test('gibt null zurück bei Tab auf Extension-Seite', async () => {
        const mock = buildChromeMock({
            activeTab: { id: 2, url: 'chrome-extension://abc/popup.html' },
            tabsError: 'Cannot access a chrome-extension:// URL',
        });
        const ctx = await getFreshPageContext(mock);
        expect(ctx).toBeNull();
    });
});

// ─── 2. buildContextSummary ───────────────────────────────────────────────────

describe('buildContextSummary', () => {
    test('null-Context → Fallback-String', () => {
        expect(buildContextSummary(null)).toBe('Kein Seitenkontext verfügbar.');
    });

    test('Context mit URL und Title → korrekte Zusammenfassung', () => {
        const summary = buildContextSummary({
            url: 'https://example.com', title: 'Example', summary: { headings: ['H1'] }
        });
        expect(summary).toContain('https://example.com');
        expect(summary).toContain('Example');
        expect(summary).toContain('H1');
    });

    test('Partial Context (nur URL) → kein Crash', () => {
        expect(() => buildContextSummary({ url: 'https://x.com' })).not.toThrow();
    });

    test('Context enthält nie "null" als String', () => {
        const summary = buildContextSummary(null);
        expect(summary).not.toContain('"null"');
        expect(summary).not.toContain('null');
    });
});

// ─── 3. buildChatPayload ─────────────────────────────────────────────────────

describe('buildChatPayload — mini_task / user_task Mapping', () => {
    const base = {
        message: 'antworte nur mit verstanden verstanden !?',
        voiceLanguage: 'de-DE',
        agentName: 'speed',
        chatHistory: [],
    };

    test('user_task enthält den echten User-Prompt', () => {
        const payload = buildChatPayload({ ...base, freshContext: null });
        expect(payload.user_task).toBe('antworte nur mit verstanden verstanden !?');
    });

    test('mini_task enthält Seitenkontext — NICHT den User-Prompt', () => {
        const payload = buildChatPayload({ ...base, freshContext: null });
        expect(payload.mini_task).not.toContain('antworte nur mit verstanden');
        expect(payload.mini_task).toContain('Browser-Assistent');
    });

    test('mini_task enthält "null" NICHT als String bei fehlendem Context', () => {
        const payload = buildChatPayload({ ...base, freshContext: null });
        expect(payload.mini_task).not.toContain('null');
        expect(payload.mini_task).toContain('Kein Seitenkontext verfügbar');
    });

    test('mini_task enthält URL wenn Context vorhanden', () => {
        const payload = buildChatPayload({
            ...base,
            freshContext: { success: true, url: 'https://example.com', title: 'Test', summary: {} }
        });
        expect(payload.mini_task).toContain('https://example.com');
        expect(payload.user_task).toBe(base.message);
    });

    test('agent_name aus Settings wird korrekt übernommen', () => {
        const payload = buildChatPayload({ ...base, freshContext: null, agentName: 'analyser' });
        expect(payload.agent_name).toBe('analyser');
    });

    test('message_history wird als leeres Array weitergegeben wenn leer', () => {
        const payload = buildChatPayload({ ...base, freshContext: null, chatHistory: [] });
        expect(payload.message_history).toEqual([]);
    });

    test('message_history wird korrekt weitergegeben', () => {
        const history = [
            { role: 'user', content: 'Hallo' },
            { role: 'assistant', content: 'Hi!' },
        ];
        const payload = buildChatPayload({ ...base, freshContext: null, chatHistory: history });
        expect(payload.message_history).toHaveLength(2);
        expect(payload.message_history[0].role).toBe('user');
    });

    test('voiceLanguage wird im mini_task gesetzt', () => {
        const payload = buildChatPayload({ ...base, freshContext: null });
        expect(payload.mini_task).toContain('de-DE');
    });

    test('task_from ist immer "browser_extension"', () => {
        const payload = buildChatPayload({ ...base, freshContext: null });
        expect(payload.task_from).toBe('browser_extension');
    });
});

// ─── 4. Agent Mode — format_schema Payload ───────────────────────────────────

describe('executeStructuredAction — format_schema Payload', () => {
    function buildAgentPayload(userTask, pageContext, actionHistory) {
        const contextInfo = pageContext
            ? `Current page context: "${JSON.stringify(pageContext)}"`
            : 'Could not access the current page context.';

        const historyInfo = actionHistory?.length > 0
            ? `You have already performed these actions: ${JSON.stringify(actionHistory)}`
            : 'This is the first step.';

        const actionSchema = {
            properties: {
                thought:         { title: 'Thought', type: 'string' },
                action_type:     { enum: ['click', 'fill_form', 'navigate', 'scroll', 'extract_data', 'finish'], type: 'string' },
                target_selector: { title: 'Target Selector', type: 'string' },
                data:            { anyOf: [{ type: 'object' }, { type: 'null' }], default: null },
                continue:        { title: 'Continue Plan', type: 'boolean' },
            },
            required: ['thought', 'action_type', 'continue'],
            title: 'AgentActionSchema',
            type: 'object',
        };

        return {
            format_schema: actionSchema,
            task: `You are a web automation agent. The user's goal is: "${userTask}". ${contextInfo} ${historyInfo}`,
            agent_name: 'speed',
            auto_context: true,
        };
    }

    test('format_schema ist vorhanden und hat required-Felder', () => {
        const payload = buildAgentPayload('Log me in', null, []);
        expect(payload.format_schema).toBeDefined();
        expect(payload.format_schema.required).toContain('thought');
        expect(payload.format_schema.required).toContain('action_type');
        expect(payload.format_schema.required).toContain('continue');
    });

    test('task enthält user goal', () => {
        const payload = buildAgentPayload('Buy a ticket', null, []);
        expect(payload.task).toContain('Buy a ticket');
    });

    test('task enthält Fallback-Text wenn kein pageContext', () => {
        const payload = buildAgentPayload('test', null, []);
        expect(payload.task).toContain('Could not access the current page context');
    });

    test('task enthält pageContext wenn vorhanden', () => {
        const payload = buildAgentPayload('test', { url: 'https://shop.com', title: 'Shop' }, []);
        expect(payload.task).toContain('shop.com');
    });

    test('task enthält History wenn vorhanden', () => {
        const history = [{ action_type: 'click', target_selector: '#btn', thought: 'Clicked' }];
        const payload = buildAgentPayload('test', null, history);
        expect(payload.task).toContain('You have already performed');
        expect(payload.task).toContain('click');
    });

    test('erster Schritt ohne History: "This is the first step"', () => {
        const payload = buildAgentPayload('test', null, []);
        expect(payload.task).toContain('This is the first step');
    });

    test('action_type enum enthält "finish"', () => {
        const payload = buildAgentPayload('test', null, []);
        expect(payload.format_schema.properties.action_type.enum).toContain('finish');
    });

    test('agent_name aus Settings wird gesetzt', () => {
        const payload = buildAgentPayload('test', null, []);
        expect(payload.agent_name).toBe('speed');
    });

    test('format_schema title ist gesetzt (für create_model im Python-Handler)', () => {
        const payload = buildAgentPayload('test', null, []);
        expect(payload.format_schema.title).toBe('AgentActionSchema');
    });
});

// ─── 5. makeAPICall → Native Routing ─────────────────────────────────────────

describe('makeAPICall → Native Endpoint Mapping', () => {
    // Spiegelt background.js: endpoint → action
    function endpointToAction(endpoint) {
        return endpoint.replace(/^\/api\//, '').replace(/\//g, '_');
    }

    const cases = [
        ['/api/isaa/mini_task_completion', 'isaa_mini_task_completion'],
        ['/api/isaa/format_class',         'isaa_format_class'],
        ['/api/isaa/listAllAgents',        'isaa_listAllAgents'],
        ['/api/PasswordManager/add_password', 'PasswordManager_add_password'],
    ];

    test.each(cases)('%s → %s', (endpoint, expected) => {
        expect(endpointToAction(endpoint)).toBe(expected);
    });

    test('makeAPICall leitet Antwort korrekt weiter', async () => {
        const mock = buildChromeMock({
            runtimeResponse: { success: true, data: { result: { data: 'hello' } } },
        });
        const result = await makeAPICall(mock, '/api/isaa/mini_task_completion', 'POST', {});
        expect(result).toHaveProperty('result.data', 'hello');
    });

    test('makeAPICall gibt response.data zurück auch wenn success=false', async () => {
        const mock = buildChromeMock({
            runtimeResponse: { success: false, data: { fallback: true } },
        });
        const result = await makeAPICall(mock, '/api/test', 'GET', null);
        expect(result).toEqual({ fallback: true });
    });
});
