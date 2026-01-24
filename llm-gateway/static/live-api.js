/**
 * LLM Gateway Live Voice Client (JavaScript)
 *
 * Standalone, importierbar für Browser und Node.js
 *
 * Usage (Browser):
 *   <script src="live-api.js"></script>
 *   <script>
 *     const client = new LiveVoiceClient({ apiKey: 'sk-...' });
 *     const session = await client.createSession({ model: 'qwen3-4b' });
 *     await session.connect();
 *
 *     session.on('audio', (audioData) => playAudio(audioData));
 *     session.on('text', (text) => console.log(text));
 *
 *     await session.endTurn('Hello, how are you?');
 *   </script>
 *
 * Usage (ES Module):
 *   import { LiveVoiceClient } from './live-api.js';
 */

(function(root, factory) {
    if (typeof define === 'function' && define.amd) {
        define([], factory);
    } else if (typeof module === 'object' && module.exports) {
        module.exports = factory();
    } else {
        const exports = factory();
        root.LiveVoiceClient = exports.LiveVoiceClient;
        root.LiveSession = exports.LiveSession;
        root.AudioRecorder = exports.AudioRecorder;
        root.AudioConfig = exports.AudioConfig;
        root.WakeWordConfig = exports.WakeWordConfig;
        root.VoiceConfig = exports.VoiceConfig;
    }
}(typeof self !== 'undefined' ? self : this, function() {
    'use strict';

    // === Configuration Classes ===

    class AudioConfig {
        constructor(options = {}) {
            this.inputFormat = options.inputFormat || 'webm';
            this.outputFormat = options.outputFormat || 'opus';
            this.sampleRate = options.sampleRate || 24000;
            this.allowInterrupt = options.allowInterrupt !== false;
        }

        toJSON() {
            return {
                input_format: this.inputFormat,
                output_format: this.outputFormat,
                sample_rate: this.sampleRate,
                allow_interrupt: this.allowInterrupt
            };
        }
    }

    class WakeWordConfig {
        constructor(options = {}) {
            this.enabled = options.enabled || false;
            this.words = options.words || [];
            this.mode = options.mode || 'pre';
        }

        toJSON() {
            return {
                enabled: this.enabled,
                words: this.words,
                mode: this.mode
            };
        }
    }

    class VoiceConfig {
        constructor(options = {}) {
            this.voiceId = options.voiceId || 'default';
            this.speed = options.speed || 1.0;
            this.language = options.language || 'auto';
        }

        toJSON() {
            return {
                voice_id: this.voiceId,
                speed: this.speed,
                language: this.language
            };
        }
    }

    class LLMConfig {
        constructor(options = {}) {
            this.model = options.model;
            this.systemPrompt = options.systemPrompt || 'Du bist ein hilfreicher Assistent.';
            this.tools = options.tools || [];
            this.historyLength = options.historyLength || 20;
            this.temperature = options.temperature || 0.7;
            this.maxTokens = options.maxTokens || 1024;
        }

        toJSON() {
            return {
                model: this.model,
                system_prompt: this.systemPrompt,
                tools: this.tools,
                history_length: this.historyLength,
                temperature: this.temperature,
                max_tokens: this.maxTokens
            };
        }
    }

    // === Event Emitter ===

    class EventEmitter {
        constructor() {
            this._events = {};
        }

        on(event, listener) {
            if (!this._events[event]) {
                this._events[event] = [];
            }
            this._events[event].push(listener);
            return this;
        }

        off(event, listener) {
            if (!this._events[event]) return this;
            this._events[event] = this._events[event].filter(l => l !== listener);
            return this;
        }

        emit(event, ...args) {
            if (!this._events[event]) return false;
            this._events[event].forEach(listener => {
                try {
                    listener.apply(this, args);
                } catch (e) {
                    console.error('Event listener error:', e);
                }
            });
            return true;
        }

        once(event, listener) {
            const onceWrapper = (...args) => {
                this.off(event, onceWrapper);
                listener.apply(this, args);
            };
            return this.on(event, onceWrapper);
        }
    }

    // === Live Session ===

    class LiveSession extends EventEmitter {
        constructor(token, websocketUrl, expiresAt, apiKey) {
            super();
            this.token = token;
            this.websocketUrl = websocketUrl;
            this.expiresAt = expiresAt;
            this._apiKey = apiKey;
            this._ws = null;
            this._connected = false;
            this._audioContext = null;
            this._audioQueue = [];
            this._isPlaying = false;
        }

        get connected() {
            return this._connected;
        }

        async connect() {
            if (this._connected) return;

            return new Promise((resolve, reject) => {
                let wsUrl = this.websocketUrl;
                if (!wsUrl.startsWith('ws')) {
                    wsUrl = wsUrl.replace('http://', 'ws://').replace('https://', 'wss://');
                }

                this._ws = new WebSocket(wsUrl);

                this._ws.onopen = () => {
                    this._connected = true;
                };

                this._ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        this._handleMessage(data);
                        if (data.type === 'ready') {
                            resolve();
                        }
                    } catch (e) {
                        console.error('Message parse error:', e);
                    }
                };

                this._ws.onerror = (error) => {
                    this.emit('error', { message: 'WebSocket error' });
                    reject(error);
                };

                this._ws.onclose = (event) => {
                    this._connected = false;
                    this.emit('close', { code: event.code, reason: event.reason });
                };

                setTimeout(() => {
                    if (!this._connected) {
                        reject(new Error('Connection timeout'));
                    }
                }, 10000);
            });
        }

        disconnect() {
            if (this._ws) {
                this._ws.close(1000, 'Client disconnect');
                this._ws = null;
            }
            this._connected = false;
            this._stopAudioPlayback();
        }

        _handleMessage(data) {
            switch (data.type) {
                case 'ready':
                    this.emit('ready', data.config);
                    break;
                case 'transcript':
                    this.emit('transcript', {
                        text: data.text,
                        isInput: data.is_input,
                        isFinal: data.is_final
                    });
                    break;
                case 'text':
                    this.emit('text', {
                        text: data.text,
                        isFinal: data.is_final
                    });
                    break;
                case 'audio':
                    const audioBytes = this._base64ToBytes(data.audio);
                    this.emit('audio', {
                        audio: audioBytes,
                        format: data.format,
                        text: data.text
                    });
                    this._queueAudio(audioBytes, data.format);
                    break;
                case 'turn_complete':
                    this.emit('turnComplete', {
                        inputTranscript: data.input_transcript,
                        outputTranscript: data.output_transcript,
                        toolsCalled: data.tools_called || [],
                        interrupted: data.interrupted || false,
                        usage: data.usage || {},
                        cost: data.cost || 0,
                        latencyMs: data.latency_ms || 0
                    });
                    break;
                case 'interrupted':
                    this.emit('interrupted', { partialOutput: data.partial_output });
                    this._stopAudioPlayback();
                    break;
                case 'wake_word_required':
                    this.emit('wakeWordRequired', { message: data.message });
                    break;
                case 'error':
                    this.emit('error', { message: data.message });
                    break;
                case 'pong':
                    break;
            }
        }

        sendAudio(audioData, transcript = '', isFinal = false) {
            if (!this._connected) throw new Error('Not connected');
            const base64Audio = audioData ? this._bytesToBase64(audioData) : '';
            this._ws.send(JSON.stringify({
                type: 'audio',
                audio: base64Audio,
                transcript: transcript,
                is_final: isFinal
            }));
        }

        endTurn(transcript) {
            if (!this._connected) throw new Error('Not connected');
            this._ws.send(JSON.stringify({
                type: 'end_turn',
                transcript: transcript
            }));
        }

        interrupt() {
            if (!this._connected) return;
            this._ws.send(JSON.stringify({ type: 'interrupt' }));
            this._stopAudioPlayback();
        }

        ping() {
            if (!this._connected) return;
            this._ws.send(JSON.stringify({ type: 'ping' }));
        }

        close() {
            if (!this._connected) return;
            this._ws.send(JSON.stringify({ type: 'close' }));
            this.disconnect();
        }

        _getAudioContext() {
            if (!this._audioContext) {
                this._audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 24000
                });
            }
            return this._audioContext;
        }

        _queueAudio(audioBytes, format) {
            this._audioQueue.push({ bytes: audioBytes, format });
            if (!this._isPlaying) {
                this._playNextAudio();
            }
        }

        async _playNextAudio() {
            if (this._audioQueue.length === 0) {
                this._isPlaying = false;
                return;
            }
            this._isPlaying = true;
            const { bytes } = this._audioQueue.shift();
            try {
                const ctx = this._getAudioContext();
                if (ctx.state === 'suspended') await ctx.resume();
                const audioBuffer = await ctx.decodeAudioData(bytes.buffer.slice(0));
                const source = ctx.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(ctx.destination);
                source.onended = () => this._playNextAudio();
                source.start();
            } catch (e) {
                console.error('Audio playback error:', e);
                this._playNextAudio();
            }
        }

        _stopAudioPlayback() {
            this._audioQueue = [];
            this._isPlaying = false;
        }

        _base64ToBytes(base64) {
            const binary = atob(base64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) {
                bytes[i] = binary.charCodeAt(i);
            }
            return bytes;
        }

        _bytesToBase64(bytes) {
            let binary = '';
            for (let i = 0; i < bytes.length; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            return btoa(binary);
        }

        async converse(text) {
            return new Promise((resolve, reject) => {
                const cleanup = () => {
                    this.off('turnComplete', onComplete);
                    this.off('interrupted', onInterrupted);
                    this.off('error', onError);
                };
                const onComplete = (data) => { cleanup(); resolve(data); };
                const onInterrupted = (data) => { cleanup(); resolve({ ...data, interrupted: true }); };
                const onError = (data) => { cleanup(); reject(new Error(data.message)); };
                this.on('turnComplete', onComplete);
                this.on('interrupted', onInterrupted);
                this.on('error', onError);
                this.endTurn(text);
            });
        }
    }

    // === Audio Recorder ===

    class AudioRecorder extends EventEmitter {
        constructor(options = {}) {
            super();
            this.sampleRate = options.sampleRate || 16000;
            this.vadEnabled = options.vadEnabled !== false;
            this.vadThreshold = options.vadThreshold || 0.02;
            this.vadSilenceMs = options.vadSilenceMs || 1500;
            this._mediaRecorder = null;
            this._stream = null;
            this._chunks = [];
            this._isRecording = false;
            this._analyser = null;
            this._audioContext = null;
            this._vadTimeout = null;
            this._isSpeaking = false;
        }

        get isRecording() { return this._isRecording; }
        get isSpeaking() { return this._isSpeaking; }

        async init() {
            try {
                this._stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: this.sampleRate,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    }
                });
                this._audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const source = this._audioContext.createMediaStreamSource(this._stream);
                this._analyser = this._audioContext.createAnalyser();
                this._analyser.fftSize = 256;
                source.connect(this._analyser);
                return true;
            } catch (e) {
                this.emit('error', { message: e.message });
                return false;
            }
        }

        start() {
            if (this._isRecording || !this._stream) return;
            this._chunks = [];
            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus' : 'audio/webm';
            this._mediaRecorder = new MediaRecorder(this._stream, {
                mimeType,
                audioBitsPerSecond: 64000
            });
            this._mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) this._chunks.push(e.data);
            };
            this._mediaRecorder.onstop = () => {
                const blob = new Blob(this._chunks, { type: mimeType });
                this.emit('audio', { blob, duration: this._chunks.length * 100 });
            };
            this._mediaRecorder.start(100);
            this._isRecording = true;
            this.emit('start');
            if (this.vadEnabled) this._startVAD();
        }

        stop() {
            if (!this._isRecording) return;
            this._isRecording = false;
            this._stopVAD();
            if (this._mediaRecorder && this._mediaRecorder.state !== 'inactive') {
                this._mediaRecorder.stop();
            }
            this.emit('stop');
        }

        getLevel() {
            if (!this._analyser) return 0;
            const data = new Uint8Array(this._analyser.frequencyBinCount);
            this._analyser.getByteFrequencyData(data);
            let sum = 0;
            for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
            return Math.sqrt(sum / data.length) / 255;
        }

        _startVAD() {
            const checkLevel = () => {
                if (!this._isRecording) return;
                const level = this.getLevel();
                this.emit('level', { level });
                const isSpeaking = level > this.vadThreshold;
                if (isSpeaking && !this._isSpeaking) {
                    this._isSpeaking = true;
                    this.emit('speechStart');
                    clearTimeout(this._vadTimeout);
                } else if (!isSpeaking && this._isSpeaking) {
                    clearTimeout(this._vadTimeout);
                    this._vadTimeout = setTimeout(() => {
                        if (!this._isSpeaking) return;
                        this._isSpeaking = false;
                        this.emit('speechEnd');
                    }, this.vadSilenceMs);
                }
                requestAnimationFrame(checkLevel);
            };
            requestAnimationFrame(checkLevel);
        }

        _stopVAD() {
            clearTimeout(this._vadTimeout);
            this._isSpeaking = false;
        }

        destroy() {
            this.stop();
            if (this._stream) this._stream.getTracks().forEach(t => t.stop());
            if (this._audioContext) this._audioContext.close();
        }
    }

    // === Live Voice Client ===

    class LiveVoiceClient {
        constructor(options = {}) {
            this.apiKey = options.apiKey || '';
            this.baseUrl = (options.baseUrl || 'http://localhost:4000').replace(/\/$/, '');
        }

        async createSession(options = {}) {
            const payload = {
                audio_config: (options.audioConfig || new AudioConfig()).toJSON(),
                wake_word_config: (options.wakeWordConfig || new WakeWordConfig()).toJSON(),
                voice_config: (options.voiceConfig || new VoiceConfig()).toJSON(),
                llm_config: new LLMConfig({
                    model: options.model,
                    systemPrompt: options.systemPrompt,
                    tools: options.tools,
                    historyLength: options.historyLength,
                    temperature: options.temperature,
                    maxTokens: options.maxTokens
                }).toJSON(),
                session_ttl_minutes: options.sessionTtlMinutes || 15
            };

            const response = await fetch(`${this.baseUrl}/v1/audio/live`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: 'Request failed' }));
                throw new Error(error.detail || `HTTP ${response.status}`);
            }

            const data = await response.json();
            let wsUrl = data.websocket_url;
            if (!wsUrl.startsWith('ws')) {
                wsUrl = this.baseUrl.replace('http', 'ws') + wsUrl;
            }

            return new LiveSession(data.session_token, wsUrl, data.expires_at, this.apiKey);
        }

        async closeSession(sessionToken) {
            const response = await fetch(`${this.baseUrl}/v1/audio/live/${sessionToken}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${this.apiKey}` }
            });
            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: 'Request failed' }));
                throw new Error(error.detail || `HTTP ${response.status}`);
            }
            return response.json();
        }

        async getSessionInfo(sessionToken) {
            const response = await fetch(`${this.baseUrl}/v1/audio/live/${sessionToken}`, {
                headers: { 'Authorization': `Bearer ${this.apiKey}` }
            });
            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: 'Request failed' }));
                throw new Error(error.detail || `HTTP ${response.status}`);
            }
            return response.json();
        }

        async textToSpeech(text, options = {}) {
            const response = await fetch(`${this.baseUrl}/v1/audio/speech`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: 'tts',
                    input: text,
                    voice: options.voice || 'default',
                    speed: options.speed || 1.0,
                    response_format: options.responseFormat || 'opus'
                })
            });
            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: 'Request failed' }));
                throw new Error(error.detail || `HTTP ${response.status}`);
            }
            return response.blob();
        }

        createRecorder(options = {}) {
            return new AudioRecorder(options);
        }
    }

    return {
        LiveVoiceClient,
        LiveSession,
        AudioRecorder,
        AudioConfig,
        WakeWordConfig,
        VoiceConfig,
        LLMConfig
    };
}));
