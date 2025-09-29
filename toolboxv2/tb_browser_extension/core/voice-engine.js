// ToolBox Browser Extension - Advanced Voice Engine
// Enhanced voice I/O system with ISAA integration

class TBVoiceEngine {
    constructor() {
        this.isListening = false;
        this.isSpeaking = false;
        this.recognition = null;
        this.synthesis = null;
        this.audioContext = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.voiceCommands = new Map();
        this.settings = {
            language: 'en-US',
            voiceIndex: 0,
            speechRate: 1.0,
            speechPitch: 1.0,
            speechVolume: 1.0,
            continuousListening: false,
            noiseReduction: true,
            wakeWords: ['toolbox', 'isaa', 'computer', 'assistant']
        };

        this.init();
    }

    async init() {
        try {
            await this.initializeSpeechRecognition();
            await this.initializeSpeechSynthesis();
            await this.initializeAudioContext();
            await this.loadSettings();
            this.setupVoiceCommands();

            TBUtils.info('VoiceEngine', 'Voice engine initialized successfully');
        } catch (error) {
            TBUtils.handleError('VoiceEngine', error);
        }
    }

    async initializeSpeechRecognition() {
        if (!TBUtils.hasFeature('speechRecognition')) {
            throw new Error('Speech recognition not supported in this browser');
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();

        this.recognition.continuous = false;
        this.recognition.interimResults = true;
        this.recognition.lang = this.settings.language;
        this.recognition.maxAlternatives = 3;

        this.recognition.onstart = () => {
            this.isListening = true;
            this.showVoiceIndicator('listening');
            TBUtils.info('VoiceEngine', 'Speech recognition started');
        };

        this.recognition.onresult = (event) => {
            this.handleSpeechResult(event);
        };

        this.recognition.onerror = (event) => {
            this.isListening = false;
            this.hideVoiceIndicator();
            TBUtils.error('VoiceEngine', `Speech recognition error: ${event.error}`);
        };

        this.recognition.onend = () => {
            this.isListening = false;
            this.hideVoiceIndicator();
            TBUtils.info('VoiceEngine', 'Speech recognition ended');
        };
    }

    async initializeSpeechSynthesis() {
        if (!TBUtils.hasFeature('speechSynthesis')) {
            throw new Error('Speech synthesis not supported in this browser');
        }

        this.synthesis = window.speechSynthesis;

        // Wait for voices to load
        return new Promise((resolve) => {
            const loadVoices = () => {
                const voices = this.synthesis.getVoices();
                if (voices.length > 0) {
                    TBUtils.info('VoiceEngine', `Loaded ${voices.length} voices`);
                    resolve();
                } else {
                    setTimeout(loadVoices, 100);
                }
            };

            if (this.synthesis.onvoiceschanged !== undefined) {
                this.synthesis.onvoiceschanged = loadVoices;
            } else {
                loadVoices();
            }
        });
    }

    async initializeAudioContext() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            TBUtils.info('VoiceEngine', 'Audio context initialized');
        } catch (error) {
            TBUtils.warn('VoiceEngine', 'Could not initialize audio context', error);
        }
    }

    async loadSettings() {
        const stored = await TBUtils.getStorage([
            'voice_language',
            'voice_index',
            'speech_rate',
            'speech_pitch',
            'speech_volume',
            'continuous_listening',
            'noise_reduction',
            'wake_words'
        ]);

        this.settings = {
            ...this.settings,
            language: stored.voice_language || this.settings.language,
            voiceIndex: stored.voice_index || this.settings.voiceIndex,
            speechRate: stored.speech_rate || this.settings.speechRate,
            speechPitch: stored.speech_pitch || this.settings.speechPitch,
            speechVolume: stored.speech_volume || this.settings.speechVolume,
            continuousListening: stored.continuous_listening || this.settings.continuousListening,
            noiseReduction: stored.noise_reduction || this.settings.noiseReduction,
            wakeWords: stored.wake_words || this.settings.wakeWords
        };

        if (this.recognition) {
            this.recognition.lang = this.settings.language;
            this.recognition.continuous = this.settings.continuousListening;
        }
    }

    setupVoiceCommands() {
        // Core voice commands
        this.voiceCommands.set('open toolbox', () => this.executeCommand('toggle-panel'));
        this.voiceCommands.set('close toolbox', () => this.executeCommand('close-panel'));
        this.voiceCommands.set('search page', (text) => this.executeCommand('search-page', { text }));
        this.voiceCommands.set('analyze page', () => this.executeCommand('analyze-page'));
        this.voiceCommands.set('auto login', () => this.executeCommand('auto-login'));
        this.voiceCommands.set('generate password', () => this.executeCommand('generate-password'));
        this.voiceCommands.set('take screenshot', () => this.executeCommand('screenshot'));
        this.voiceCommands.set('scroll up', () => this.executeCommand('scroll', { direction: 'up' }));
        this.voiceCommands.set('scroll down', () => this.executeCommand('scroll', { direction: 'down' }));
        this.voiceCommands.set('go back', () => this.executeCommand('navigate', { direction: 'back' }));
        this.voiceCommands.set('go forward', () => this.executeCommand('navigate', { direction: 'forward' }));
        this.voiceCommands.set('refresh page', () => this.executeCommand('refresh'));
        this.voiceCommands.set('stop listening', () => this.stopListening());

        // ISAA integration commands
        this.voiceCommands.set('ask isaa', (text) => this.executeCommand('isaa-query', { text }));
        this.voiceCommands.set('isaa help', () => this.executeCommand('isaa-help'));
        this.voiceCommands.set('smart search', (text) => this.executeCommand('smart-search', { text }));
    }

    async startListening() {
        if (this.isListening || !this.recognition) return;

        try {
            // Request microphone permission
            await navigator.mediaDevices.getUserMedia({ audio: true });

            this.recognition.start();
            TBUtils.info('VoiceEngine', 'Started listening for voice commands');
        } catch (error) {
            TBUtils.handleError('VoiceEngine', error);
            this.showError('Microphone access denied or not available');
        }
    }

    stopListening() {
        if (!this.isListening || !this.recognition) return;

        this.recognition.stop();
        TBUtils.info('VoiceEngine', 'Stopped listening for voice commands');
    }

    async speak(text, options = {}) {
        if (this.isSpeaking) {
            this.synthesis.cancel();
        }

        const utterance = new SpeechSynthesisUtterance(text);
        const voices = this.synthesis.getVoices();

        if (voices[this.settings.voiceIndex]) {
            utterance.voice = voices[this.settings.voiceIndex];
        }

        utterance.rate = options.rate || this.settings.speechRate;
        utterance.pitch = options.pitch || this.settings.speechPitch;
        utterance.volume = options.volume || this.settings.speechVolume;

        utterance.onstart = () => {
            this.isSpeaking = true;
            this.showVoiceIndicator('speaking');
        };

        utterance.onend = () => {
            this.isSpeaking = false;
            this.hideVoiceIndicator();
        };

        utterance.onerror = (event) => {
            this.isSpeaking = false;
            this.hideVoiceIndicator();
            TBUtils.error('VoiceEngine', `Speech synthesis error: ${event.error}`);
        };

        this.synthesis.speak(utterance);
        TBUtils.info('VoiceEngine', `Speaking: ${text.substring(0, 50)}...`);
    }

    handleSpeechResult(event) {
        let finalTranscript = '';
        let interimTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;

            if (event.results[i].isFinal) {
                finalTranscript += transcript;
            } else {
                interimTranscript += transcript;
            }
        }

        // Update UI with interim results
        if (interimTranscript) {
            this.updateVoiceIndicator(interimTranscript);
        }

        // Process final transcript
        if (finalTranscript) {
            this.processVoiceCommand(finalTranscript.trim().toLowerCase());
        }
    }

    processVoiceCommand(transcript) {
        TBUtils.info('VoiceEngine', `Processing voice command: "${transcript}"`);

        // Check for wake words first
        const hasWakeWord = this.settings.wakeWords.some(word =>
            transcript.includes(word.toLowerCase())
        );

        if (!hasWakeWord && !this.settings.continuousListening) {
            TBUtils.info('VoiceEngine', 'No wake word detected, ignoring command');
            return;
        }

        // Find matching command
        let commandExecuted = false;

        for (const [command, handler] of this.voiceCommands) {
            if (transcript.includes(command)) {
                const remainingText = transcript.replace(command, '').trim();
                handler(remainingText);
                commandExecuted = true;
                break;
            }
        }

        if (!commandExecuted) {
            // Send to ISAA for processing
            this.executeCommand('isaa-query', { text: transcript });
        }
    }

    async executeCommand(command, data = {}) {
        try {
            // Send command to content script or background
            if (typeof chrome !== 'undefined' && chrome.runtime) {
                const response = await chrome.runtime.sendMessage({
                    type: 'TB_VOICE_COMMAND',
                    command,
                    data,
                    timestamp: Date.now()
                });

                if (response && response.speak) {
                    await this.speak(response.speak);
                }
            }
        } catch (error) {
            TBUtils.handleError('VoiceEngine', error);
        }
    }

    showVoiceIndicator(state) {
        // Implementation will be in UI manager
        if (window.tbUIManager) {
            window.tbUIManager.showVoiceIndicator(state);
        }
    }

    updateVoiceIndicator(text) {
        if (window.tbUIManager) {
            window.tbUIManager.updateVoiceIndicator(text);
        }
    }

    hideVoiceIndicator() {
        if (window.tbUIManager) {
            window.tbUIManager.hideVoiceIndicator();
        }
    }

    showError(message) {
        if (window.tbUIManager) {
            window.tbUIManager.showNotification(message, 'error');
        }
    }

    // Public API
    async toggle() {
        if (this.isListening) {
            this.stopListening();
        } else {
            await this.startListening();
        }
    }

    getAvailableVoices() {
        return this.synthesis ? this.synthesis.getVoices() : [];
    }

    async updateSettings(newSettings) {
        this.settings = { ...this.settings, ...newSettings };
        await TBUtils.setStorage({
            voice_language: this.settings.language,
            voice_index: this.settings.voiceIndex,
            speech_rate: this.settings.speechRate,
            speech_pitch: this.settings.speechPitch,
            speech_volume: this.settings.speechVolume,
            continuous_listening: this.settings.continuousListening,
            noise_reduction: this.settings.noiseReduction,
            wake_words: this.settings.wakeWords
        });

        if (this.recognition) {
            this.recognition.lang = this.settings.language;
            this.recognition.continuous = this.settings.continuousListening;
        }
    }
}

// Export for global use
if (typeof window !== 'undefined') {
    window.TBVoiceEngine = TBVoiceEngine;
}
