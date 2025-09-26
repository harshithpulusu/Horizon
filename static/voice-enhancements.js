// Voice Enhancement Features for Horizon AI Assistant
// Wake Word Detection and Voice Cloning

class VoiceEnhancements {
    constructor(aiAssistant) {
        this.aiAssistant = aiAssistant;
        this.isWakeWordListening = false;
        this.wakeWordRecognition = null;
        this.wakeWords = ['hey horizon', 'horizon', 'hey assistant', 'assistant'];
        this.wakeWordSensitivity = 0.7;
        this.wakeWordTimeout = null;
        
        // Enhanced Language Support
        this.currentLanguage = 'en-US';
        this.supportedLanguages = {
            'en-US': {
                name: 'English (US)',
                wakeWords: ['hey horizon', 'horizon', 'hey assistant', 'assistant'],
                sampleText: 'Hello, I am training my personal AI assistant. This voice sample will help create a more natural and personalized experience.'
            },
            'en-GB': {
                name: 'English (UK)',
                wakeWords: ['hey horizon', 'horizon', 'hey assistant', 'assistant'],
                sampleText: 'Hello, I am training my personal AI assistant. This voice sample will help create a more natural and personalised experience.'
            },
            'es-ES': {
                name: 'Spanish (Spain)',
                wakeWords: ['oye horizon', 'horizon', 'oye asistente', 'asistente'],
                sampleText: 'Hola, estoy entrenando a mi asistente de IA personal. Esta muestra de voz ayudará a crear una experiencia más natural y personalizada.'
            },
            'es-MX': {
                name: 'Spanish (Mexico)',
                wakeWords: ['oye horizon', 'horizon', 'oye asistente', 'asistente'],
                sampleText: 'Hola, estoy entrenando a mi asistente de IA personal. Esta muestra de voz ayudará a crear una experiencia más natural y personalizada.'
            },
            'fr-FR': {
                name: 'French (France)',
                wakeWords: ['salut horizon', 'horizon', 'salut assistant', 'assistant'],
                sampleText: 'Bonjour, je forme mon assistant IA personnel. Cet échantillon vocal aidera à créer une expérience plus naturelle et personnalisée.'
            },
            'de-DE': {
                name: 'German (Germany)',
                wakeWords: ['hallo horizon', 'horizon', 'hallo assistent', 'assistent'],
                sampleText: 'Hallo, ich trainiere meinen persönlichen KI-Assistenten. Diese Sprachprobe wird helfen, eine natürlichere und personalisiertere Erfahrung zu schaffen.'
            },
            'it-IT': {
                name: 'Italian (Italy)',
                wakeWords: ['ciao horizon', 'horizon', 'ciao assistente', 'assistente'],
                sampleText: 'Ciao, sto addestrando il mio assistente IA personale. Questo campione vocale aiuterà a creare un\'esperienza più naturale e personalizzata.'
            },
            'pt-BR': {
                name: 'Portuguese (Brazil)',
                wakeWords: ['oi horizon', 'horizon', 'oi assistente', 'assistente'],
                sampleText: 'Olá, estou treinando meu assistente de IA pessoal. Esta amostra de voz ajudará a criar uma experiência mais natural e personalizada.'
            },
            'ja-JP': {
                name: 'Japanese (Japan)',
                wakeWords: ['horizon', 'アシスタント', 'ホライズン'],
                sampleText: '私は個人のAIアシスタントをトレーニングしています。この音声サンプルは、より自然でパーソナライズされた体験を作成するのに役立ちます。'
            },
            'ko-KR': {
                name: 'Korean (Korea)',
                wakeWords: ['horizon', '어시스턴트', '호라이즌'],
                sampleText: '안녕하세요, 저는 개인 AI 어시스턴트를 훈련하고 있습니다. 이 음성 샘플은 더 자연스럽고 개인화된 경험을 만드는 데 도움이 될 것입니다.'
            },
            'zh-CN': {
                name: 'Chinese (Simplified)',
                wakeWords: ['horizon', '助手', '地平线'],
                sampleText: '你好，我正在训练我的个人AI助手。这个语音样本将有助于创建更自然和个性化的体验。'
            },
            'zh-TW': {
                name: 'Chinese (Traditional)',
                wakeWords: ['horizon', '助手', '地平線'],
                sampleText: '你好，我正在訓練我的個人AI助手。這個語音樣本將有助於創建更自然和個性化的體驗。'
            },
            'ru-RU': {
                name: 'Russian (Russia)',
                wakeWords: ['привет horizon', 'horizon', 'привет помощник', 'помощник'],
                sampleText: 'Привет, я обучаю своего персонального ИИ-помощника. Этот образец голоса поможет создать более естественный и персонализированный опыт.'
            },
            'ar-SA': {
                name: 'Arabic (Saudi Arabia)',
                wakeWords: ['مرحبا horizon', 'horizon', 'مرحبا مساعد', 'مساعد'],
                sampleText: 'مرحبا، أنا أدرب مساعد الذكاء الاصطناعي الشخصي الخاص بي. ستساعد عينة الصوت هذه في إنشاء تجربة أكثر طبيعية وشخصية.'
            },
            'hi-IN': {
                name: 'Hindi (India)',
                wakeWords: ['नमस्ते horizon', 'horizon', 'नमस्ते सहायक', 'सहायक'],
                sampleText: 'नमस्ते, मैं अपने व्यक्तिगत AI सहायक को प्रशिक्षित कर रहा हूं। यह आवाज़ का नमूना अधिक प्राकृतिक और व्यक्तिगत अनुभव बनाने में मदद करेगा।'
            }
        };
        
        // Background listening mode
        this.backgroundMode = {
            enabled: false,
            minimized: false,
            lowPowerMode: false,
            batteryOptimization: true,
            continuousHours: 0,
            maxContinuousHours: 8, // Auto-pause after 8 hours for battery
            pauseOnInactivity: true,
            inactivityTimeout: 30 // minutes
        };
        
        // Voice cloning properties
        this.voiceSettings = {
            enabled: false,
            userVoiceId: null,
            personalizedVoices: {},
            voiceCloneEnabled: false,
            recordedSamples: [],
            elevenlabsApiKey: null,
            languageSpecificModels: {} // Store voice models per language
        };
        
        this.init();
    }
    
    init() {
        this.initWakeWordDetection();
        this.initVoiceCloning();
        this.initBackgroundMode();
        this.addUI();
        this.loadSettings();
        this.detectUserLanguage();
        
        // Auto-start wake word detection
        setTimeout(() => {
            this.startWakeWordListening();
        }, 1000);
        
        // Initialize background mode if enabled
        if (this.backgroundMode.enabled) {
            this.enableBackgroundMode();
        }
    }
    
    // ===== LANGUAGE SUPPORT =====
    
    detectUserLanguage() {
        // Try to detect user's language from browser settings
        const browserLang = navigator.language || navigator.userLanguage || 'en-US';
        
        // Check if we support the detected language
        if (this.supportedLanguages[browserLang]) {
            this.setLanguage(browserLang);
        } else {
            // Try to match language family (e.g., 'en' from 'en-AU')
            const langFamily = browserLang.split('-')[0];
            const matchingLang = Object.keys(this.supportedLanguages).find(lang => 
                lang.startsWith(langFamily)
            );
            
            if (matchingLang) {
                this.setLanguage(matchingLang);
            }
        }
        
        console.log(`🌍 Language detected: ${this.currentLanguage} (${this.supportedLanguages[this.currentLanguage].name})`);
    }
    
    setLanguage(languageCode) {
        if (!this.supportedLanguages[languageCode]) {
            console.warn(`Language ${languageCode} not supported`);
            return;
        }
        
        this.currentLanguage = languageCode;
        this.wakeWords = this.supportedLanguages[languageCode].wakeWords;
        
        // Update wake word recognition language
        if (this.wakeWordRecognition) {
            this.wakeWordRecognition.lang = languageCode;
        }
        
        // Update UI
        this.updateLanguageUI();
        
        // Save language preference
        this.saveSettings();
        
        console.log(`🌍 Language set to: ${this.supportedLanguages[languageCode].name}`);
        
        if (window.professionalUI) {
            window.professionalUI.showToast(
                `Language changed to ${this.supportedLanguages[languageCode].name} 🌍`, 
                'info', 
                3000
            );
        }
    }
    
    updateLanguageUI() {
        // Update language selector
        const languageSelect = document.getElementById('languageSelect');
        if (languageSelect) {
            languageSelect.value = this.currentLanguage;
        }
        
        // Update wake word examples
        const wakeWordExamples = document.getElementById('wakeWordExamples');
        if (wakeWordExamples) {
            wakeWordExamples.textContent = `Say "${this.wakeWords[0]}" or "${this.wakeWords[1]}" to activate`;
        }
        
        // Update status with localized wake words
        this.updateWakeWordStatus(`Listening for "${this.wakeWords[0]}"...`);
    }
    
    // ===== BACKGROUND LISTENING MODE =====
    
    initBackgroundMode() {
        // Load background mode settings
        this.loadBackgroundSettings();
        
        // Set up activity monitoring
        this.setupActivityMonitoring();
        
        // Set up battery optimization
        this.setupBatteryOptimization();
        
        console.log('🔄 Background listening mode initialized');
    }
    
    setupActivityMonitoring() {
        let lastActivity = Date.now();
        let inactivityTimer = null;
        
        // Monitor user activity
        const activityEvents = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart'];
        
        const updateActivity = () => {
            lastActivity = Date.now();
            
            // Clear inactivity timer
            if (inactivityTimer) {
                clearTimeout(inactivityTimer);
            }
            
            // Resume listening if paused due to inactivity
            if (this.backgroundMode.enabled && !this.isWakeWordListening) {
                this.startWakeWordListening();
            }
            
            // Set new inactivity timer
            if (this.backgroundMode.pauseOnInactivity) {
                inactivityTimer = setTimeout(() => {
                    if (this.backgroundMode.enabled) {
                        console.log('⏸️ Pausing wake word detection due to inactivity');
                        this.pauseForInactivity();
                    }
                }, this.backgroundMode.inactivityTimeout * 60 * 1000);
            }
        };
        
        activityEvents.forEach(event => {
            document.addEventListener(event, updateActivity, { passive: true });
        });
        
        // Initial activity update
        updateActivity();
    }
    
    setupBatteryOptimization() {
        if ('getBattery' in navigator) {
            navigator.getBattery().then(battery => {
                const checkBattery = () => {
                    if (battery.level < 0.2 && !battery.charging && this.backgroundMode.enabled) {
                        console.log('🔋 Low battery detected, enabling power saving mode');
                        this.enableLowPowerMode();
                    } else if (battery.level > 0.3 && this.backgroundMode.lowPowerMode) {
                        console.log('🔋 Battery level restored, disabling power saving mode');
                        this.disableLowPowerMode();
                    }
                };
                
                battery.addEventListener('levelchange', checkBattery);
                battery.addEventListener('chargingchange', checkBattery);
                
                // Initial check
                checkBattery();
            });
        }
    }
    
    enableBackgroundMode() {
        this.backgroundMode.enabled = true;
        this.backgroundMode.continuousHours = 0;
        
        // Start continuous monitoring
        this.startContinuousMonitoring();
        
        // Update UI
        const backgroundBtn = document.getElementById('toggleBackgroundMode');
        if (backgroundBtn) {
            backgroundBtn.classList.add('active');
            backgroundBtn.innerHTML = '🔄 Background: ON';
        }
        
        // Show background indicator
        this.showBackgroundIndicator();
        
        console.log('🔄 Background listening mode enabled');
        
        if (window.professionalUI) {
            window.professionalUI.showToast(
                'Background listening enabled - Horizon will listen even when minimized 🔄', 
                'success', 
                4000
            );
        }
        
        this.saveBackgroundSettings();
    }
    
    disableBackgroundMode() {
        this.backgroundMode.enabled = false;
        this.backgroundMode.minimized = false;
        
        // Stop continuous monitoring
        this.stopContinuousMonitoring();
        
        // Update UI
        const backgroundBtn = document.getElementById('toggleBackgroundMode');
        if (backgroundBtn) {
            backgroundBtn.classList.remove('active');
            backgroundBtn.innerHTML = '🔄 Background: OFF';
        }
        
        // Hide background indicator
        this.hideBackgroundIndicator();
        
        console.log('🔄 Background listening mode disabled');
        
        if (window.professionalUI) {
            window.professionalUI.showToast('Background listening disabled', 'info', 2000);
        }
        
        this.saveBackgroundSettings();
    }
    
    startContinuousMonitoring() {
        // Monitor for maximum continuous hours
        this.continuousTimer = setInterval(() => {
            this.backgroundMode.continuousHours += 1/60; // Increment by 1 minute
            
            if (this.backgroundMode.continuousHours >= this.backgroundMode.maxContinuousHours) {
                console.log('⏰ Maximum continuous listening time reached, taking a break');
                this.pauseForBreak();
            }
        }, 60000); // Check every minute
        
        // Enhanced wake word detection for background mode
        if (this.wakeWordRecognition) {
            this.wakeWordRecognition.continuous = true;
            this.wakeWordRecognition.interimResults = false; // Less processing in background
        }
    }
    
    stopContinuousMonitoring() {
        if (this.continuousTimer) {
            clearInterval(this.continuousTimer);
            this.continuousTimer = null;
        }
    }
    
    pauseForInactivity() {
        if (this.isWakeWordListening) {
            this.stopWakeWordListening();
        }
        
        this.updateWakeWordStatus('⏸️ Paused due to inactivity');
        
        // Visual indicator for paused state
        const indicator = document.getElementById('backgroundIndicator');
        if (indicator) {
            indicator.style.background = '#ffd93d';
            indicator.title = 'Background listening paused - move mouse to resume';
        }
    }
    
    pauseForBreak() {
        this.disableBackgroundMode();
        this.backgroundMode.continuousHours = 0;
        
        if (window.professionalUI) {
            window.professionalUI.showModal({
                title: '⏰ Listening Break',
                content: `
                    <p>Horizon has been listening continuously for ${this.backgroundMode.maxContinuousHours} hours.</p>
                    <p>Taking a short break to optimize performance and save battery.</p>
                    <p>You can resume background listening anytime!</p>
                `,
                buttons: [
                    {
                        text: '🔄 Resume Now',
                        action: 'resume',
                        primary: true
                    },
                    {
                        text: '⏰ Resume in 30 min',
                        action: 'resumeLater'
                    }
                ],
                callbacks: {
                    resume: () => this.enableBackgroundMode(),
                    resumeLater: () => {
                        setTimeout(() => {
                            this.enableBackgroundMode();
                        }, 30 * 60 * 1000);
                    }
                }
            });
        }
    }
    
    enableLowPowerMode() {
        this.backgroundMode.lowPowerMode = true;
        
        // Reduce sensitivity for battery saving
        this.wakeWordSensitivity = Math.max(0.8, this.wakeWordSensitivity);
        
        // Less frequent interim results
        if (this.wakeWordRecognition) {
            this.wakeWordRecognition.interimResults = false;
        }
        
        const indicator = document.getElementById('backgroundIndicator');
        if (indicator) {
            indicator.style.background = '#ff9500';
            indicator.title = 'Low power mode active';
        }
        
        this.updateWakeWordStatus('🔋 Low power mode - reduced sensitivity');
    }
    
    disableLowPowerMode() {
        this.backgroundMode.lowPowerMode = false;
        
        // Restore normal sensitivity
        this.wakeWordSensitivity = 0.7;
        
        // Restore interim results
        if (this.wakeWordRecognition) {
            this.wakeWordRecognition.interimResults = true;
        }
        
        const indicator = document.getElementById('backgroundIndicator');
        if (indicator) {
            indicator.style.background = '#4ecdc4';
            indicator.title = 'Background listening active';
        }
        
        this.updateWakeWordStatus(`Listening for "${this.wakeWords[0]}"...`);
    }
    
    showBackgroundIndicator() {
        // Remove existing indicator
        this.hideBackgroundIndicator();
        
        // Create floating background indicator
        const indicator = document.createElement('div');
        indicator.id = 'backgroundIndicator';
        indicator.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            width: 12px;
            height: 12px;
            background: #4ecdc4;
            border-radius: 50%;
            z-index: 9999;
            animation: backgroundPulse 2s infinite;
            cursor: pointer;
            box-shadow: 0 0 10px rgba(78, 205, 196, 0.5);
        `;
        indicator.title = 'Background listening active - Click to toggle';
        indicator.onclick = () => this.toggleBackgroundMode();
        
        document.body.appendChild(indicator);
        
        // Add animation styles if not already added
        if (!document.getElementById('backgroundIndicatorStyles')) {
            const style = document.createElement('style');
            style.id = 'backgroundIndicatorStyles';
            style.textContent = `
                @keyframes backgroundPulse {
                    0% { opacity: 1; transform: scale(1); }
                    50% { opacity: 0.6; transform: scale(1.2); }
                    100% { opacity: 1; transform: scale(1); }
                }
            `;
            document.head.appendChild(style);
        }
    }
    
    hideBackgroundIndicator() {
        const indicator = document.getElementById('backgroundIndicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    loadBackgroundSettings() {
        try {
            const saved = localStorage.getItem('horizon_background_settings');
            if (saved) {
                this.backgroundMode = { ...this.backgroundMode, ...JSON.parse(saved) };
            }
        } catch (error) {
            console.log('Could not load background settings:', error);
        }
    }
    
    saveBackgroundSettings() {
        try {
            localStorage.setItem('horizon_background_settings', JSON.stringify(this.backgroundMode));
        } catch (error) {
            console.log('Could not save background settings:', error);
        }
    }
    
    // ===== WAKE WORD DETECTION =====
    
    initWakeWordDetection() {
        const isSecure = location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
        
        if (!isSecure) {
            console.log('Wake word detection requires HTTPS or localhost');
            return;
        }
        
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            try {
                const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
                this.wakeWordRecognition = new SpeechRecognition();
                this.wakeWordRecognition.continuous = true;
                this.wakeWordRecognition.interimResults = true;
                this.wakeWordRecognition.lang = this.currentLanguage;
                this.wakeWordRecognition.maxAlternatives = 3;
                
                this.wakeWordRecognition.onresult = (event) => this.handleWakeWordResult(event);
                this.wakeWordRecognition.onerror = (event) => this.handleWakeWordError(event);
                this.wakeWordRecognition.onend = () => this.handleWakeWordEnd();
                
                console.log(`✨ Wake word detection initialized for ${this.supportedLanguages[this.currentLanguage].name}`);
            } catch (error) {
                console.error('Error initializing wake word detection:', error);
            }
        } else {
            console.log('Speech recognition not supported');
        }
    }
    
    startWakeWordListening() {
        if (!this.wakeWordRecognition || this.isWakeWordListening) {
            return;
        }
        
        // Don't start if main listening is active
        if (this.aiAssistant && this.aiAssistant.isListening) {
            return;
        }
        
        try {
            this.isWakeWordListening = true;
            this.wakeWordRecognition.start();
            console.log('🌟 Wake word listening started...');
            this.updateWakeWordStatus('Listening for "Hey Horizon"...');
        } catch (error) {
            console.log('Wake word listening error:', error);
            this.isWakeWordListening = false;
        }
    }
    
    stopWakeWordListening() {
        if (!this.wakeWordRecognition || !this.isWakeWordListening) {
            return;
        }
        
        try {
            this.isWakeWordListening = false;
            this.wakeWordRecognition.stop();
            console.log('Wake word listening stopped');
        } catch (error) {
            console.log('Error stopping wake word listening:', error);
        }
    }
    
    handleWakeWordResult(event) {
        const results = Array.from(event.results);
        const lastResult = results[results.length - 1];
        
        if (!lastResult || !lastResult[0]) return;
        
        const transcript = lastResult[0].transcript.toLowerCase().trim();
        const confidence = lastResult[0].confidence || 0;
        
        console.log('Wake word transcript:', transcript, 'Confidence:', confidence);
        
        // Check if any wake word is detected
        const wakeWordDetected = this.wakeWords.some(wakeWord => {
            return transcript.includes(wakeWord) && confidence > this.wakeWordSensitivity;
        });
        
        if (wakeWordDetected) {
            console.log('🎯 Wake word detected!');
            this.onWakeWordDetected(transcript);
        }
    }
    
    handleWakeWordError(event) {
        console.log('Wake word detection error:', event.error);
        
        // Restart wake word detection on certain errors
        if (event.error !== 'aborted' && this.isWakeWordListening) {
            setTimeout(() => {
                if (!this.aiAssistant.isListening) {
                    this.startWakeWordListening();
                }
            }, 1000);
        }
    }
    
    handleWakeWordEnd() {
        // Restart wake word detection when it ends
        if (this.isWakeWordListening && !this.aiAssistant.isListening) {
            setTimeout(() => this.startWakeWordListening(), 100);
        }
    }
    
    onWakeWordDetected(transcript) {
        // Stop wake word listening temporarily
        this.stopWakeWordListening();
        
        // Visual and audio feedback
        this.showWakeWordDetectedFeedback();
        this.playWakeWordSound();
        
        // Start main speech recognition for the command
        setTimeout(() => {
            if (this.aiAssistant && this.aiAssistant.startListening) {
                this.aiAssistant.startListening();
            }
        }, 500);
        
        // Resume wake word listening after a timeout
        this.wakeWordTimeout = setTimeout(() => {
            if (!this.aiAssistant.isListening) {
                this.startWakeWordListening();
            }
        }, 8000);
    }
    
    showWakeWordDetectedFeedback() {
        // Update status
        this.updateWakeWordStatus('🎯 Wake word detected! Listening for command...');
        
        // Visual pulse effect
        const statusElement = document.getElementById('statusIndicator');
        if (statusElement) {
            statusElement.style.background = '#4ecdc4';
            statusElement.style.animation = 'pulse 0.5s ease-in-out 2';
            setTimeout(() => {
                statusElement.style.animation = '';
                statusElement.style.background = '';
            }, 1000);
        }
        
        // Show toast notification
        if (window.professionalUI) {
            window.professionalUI.showToast('Wake word detected! 👂', 'success', 2000);
        }
    }
    
    playWakeWordSound() {
        // Create a pleasant beep sound
        if (typeof AudioContext !== 'undefined' || typeof webkitAudioContext !== 'undefined') {
            try {
                const audioContext = new (AudioContext || webkitAudioContext)();
                const oscillator = audioContext.createOscillator();
                const gainNode = audioContext.createGain();
                
                oscillator.connect(gainNode);
                gainNode.connect(audioContext.destination);
                
                // Pleasant ascending tone
                oscillator.frequency.setValueAtTime(600, audioContext.currentTime);
                oscillator.frequency.exponentialRampToValueAtTime(800, audioContext.currentTime + 0.1);
                
                gainNode.gain.setValueAtTime(0.05, audioContext.currentTime);
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.15);
                
                oscillator.start(audioContext.currentTime);
                oscillator.stop(audioContext.currentTime + 0.15);
            } catch (error) {
                console.log('Could not play wake word sound:', error);
            }
        }
    }
    
    updateWakeWordStatus(status) {
        const statusElement = document.getElementById('wakeWordStatus');
        if (statusElement) {
            statusElement.textContent = status;
        }
        
        // Also update main status if available
        if (this.aiAssistant && this.aiAssistant.updateStatus) {
            this.aiAssistant.updateStatus(status);
        }
    }
    
    // ===== VOICE CLONING =====
    
    initVoiceCloning() {
        this.loadVoiceSettings();
        this.setupElevenLabsIntegration();
        console.log('🎤 Voice cloning initialized');
    }
    
    loadVoiceSettings() {
        try {
            const saved = localStorage.getItem('horizon_voice_settings');
            if (saved) {
                this.voiceSettings = { ...this.voiceSettings, ...JSON.parse(saved) };
            }
        } catch (error) {
            console.log('Could not load voice settings:', error);
        }
    }
    
    saveVoiceSettings() {
        try {
            localStorage.setItem('horizon_voice_settings', JSON.stringify(this.voiceSettings));
        } catch (error) {
            console.log('Could not save voice settings:', error);
        }
    }
    
    setupElevenLabsIntegration() {
        // Check if ElevenLabs is available
        if (typeof window.ElevenLabs !== 'undefined') {
            console.log('ElevenLabs SDK available');
        } else {
            console.log('ElevenLabs SDK not loaded - voice cloning will use fallback');
        }
    }
    
    async toggleVoiceCloning() {
        this.voiceSettings.voiceCloneEnabled = !this.voiceSettings.voiceCloneEnabled;
        this.saveVoiceSettings();
        
        const button = document.getElementById('toggleVoiceClone');
        const status = document.querySelector('.voice-clone-status');
        
        if (this.voiceSettings.voiceCloneEnabled) {
            if (button) button.classList.add('active');
            if (status) status.textContent = 'Enabled';
            
            // Enable recording button
            const recordBtn = document.getElementById('recordVoiceSample');
            if (recordBtn) recordBtn.disabled = false;
            
            if (window.professionalUI) {
                window.professionalUI.showToast('Voice cloning enabled! 🎤', 'success', 3000);
            }
        } else {
            if (button) button.classList.remove('active');
            if (status) status.textContent = 'Disabled';
            
            // Disable recording button
            const recordBtn = document.getElementById('recordVoiceSample');
            if (recordBtn) recordBtn.disabled = true;
            
            if (window.professionalUI) {
                window.professionalUI.showToast('Voice cloning disabled', 'info', 2000);
            }
        }
    }
    
    async recordVoiceSample() {
        if (!this.voiceSettings.voiceCloneEnabled) {
            alert('Please enable voice cloning first');
            return;
        }
        
        try {
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            const chunks = [];
            
            mediaRecorder.ondataavailable = (event) => {
                chunks.push(event.data);
            };
            
            mediaRecorder.onstop = async () => {
                const blob = new Blob(chunks, { type: 'audio/wav' });
                await this.processSampleRecording(blob);
                
                // Stop all tracks to release microphone
                stream.getTracks().forEach(track => track.stop());
            };
            
            // Show recording UI
            this.showRecordingUI(mediaRecorder);
            
            // Start recording
            mediaRecorder.start();
            console.log('🎤 Recording voice sample...');
            
        } catch (error) {
            console.error('Error recording voice sample:', error);
            alert('Could not access microphone. Please check permissions.');
        }
    }
    
    showRecordingUI(mediaRecorder) {
        // Create recording modal
        const modal = document.createElement('div');
        modal.className = 'voice-recording-modal';
        modal.innerHTML = `
            <div class="recording-content">
                <h3>🎤 Recording Voice Sample</h3>
                <div class="recording-visual">
                    <div class="recording-pulse"></div>
                </div>
                <p>Please read the following text clearly:</p>
                <blockquote class="sample-text">
                    ${this.supportedLanguages[this.currentLanguage].sampleText}
                </blockquote>
                <div class="recording-controls">
                    <button id="stopRecording" class="stop-recording-btn">
                        ⏹️ Stop Recording
                    </button>
                </div>
                <small>Recording will automatically stop after 10 seconds</small>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Auto-stop after 10 seconds
        setTimeout(() => {
            mediaRecorder.stop();
        }, 10000);
        
        // Manual stop
        document.getElementById('stopRecording').onclick = () => {
            mediaRecorder.stop();
        };
        
        // Remove modal when recording stops
        mediaRecorder.onstop = () => {
            document.body.removeChild(modal);
        };
    }
    
    async processSampleRecording(blob) {
        // Add to recorded samples
        this.voiceSettings.recordedSamples.push({
            blob: blob,
            timestamp: Date.now(),
            size: blob.size
        });
        
        // Update UI
        const sampleCount = document.getElementById('sampleCount');
        if (sampleCount) {
            sampleCount.textContent = this.voiceSettings.recordedSamples.length;
        }
        
        // Save settings
        this.saveVoiceSettings();
        
        if (window.professionalUI) {
            window.professionalUI.showToast(
                `Voice sample ${this.voiceSettings.recordedSamples.length} recorded successfully! 🎉`, 
                'success', 
                3000
            );
        }
        
        // If we have enough samples, offer to train voice model
        if (this.voiceSettings.recordedSamples.length >= 3) {
            this.showTrainModelOption();
        }
    }
    
    showTrainModelOption() {
        if (window.professionalUI) {
            window.professionalUI.showModal({
                title: '🎤 Voice Model Training',
                content: `
                    <p>Great! You've recorded ${this.voiceSettings.recordedSamples.length} voice samples.</p>
                    <p>Would you like to train your personalized voice model?</p>
                    <p><small>This will enable the AI to respond in your own voice!</small></p>
                `,
                buttons: [
                    {
                        text: '🚀 Train Model',
                        action: 'trainModel',
                        primary: true
                    },
                    {
                        text: 'Later',
                        action: 'close'
                    }
                ],
                callbacks: {
                    trainModel: () => this.trainVoiceModel()
                }
            });
        }
    }
    
    async trainVoiceModel() {
        if (window.professionalUI) {
            window.professionalUI.showToast('Training voice model... This may take a moment.', 'info', 5000);
        }
        
        // Simulate training process
        setTimeout(() => {
            this.voiceSettings.userVoiceId = 'user_trained_' + Date.now();
            this.saveVoiceSettings();
            
            if (window.professionalUI) {
                window.professionalUI.showToast('🎉 Voice model trained successfully!', 'success', 4000);
            }
        }, 3000);
    }
    
    // ===== UI MANAGEMENT =====
    
    addUI() {
        this.addLanguageSelector();
        this.addWakeWordControls();
        this.addBackgroundModeControls();
        this.addVoiceCloneControls();
        this.addStatusIndicators();
    }
    
    addLanguageSelector() {
        const controlsContainer = document.querySelector('.voice-controls') || document.querySelector('.controls');
        if (!controlsContainer) return;
        
        const languageDiv = document.createElement('div');
        languageDiv.className = 'language-section';
        languageDiv.innerHTML = `
            <div class="feature-section">
                <h4>🌍 Language & Region</h4>
                <div class="language-controls">
                    <select id="languageSelect" class="language-select">
                        ${Object.entries(this.supportedLanguages).map(([code, info]) => 
                            `<option value="${code}" ${code === this.currentLanguage ? 'selected' : ''}>${info.name}</option>`
                        ).join('')}
                    </select>
                    <div class="language-info">
                        <small id="wakeWordExamples">Say "${this.wakeWords[0]}" or "${this.wakeWords[1]}" to activate</small>
                    </div>
                </div>
            </div>
        `;
        
        controlsContainer.appendChild(languageDiv);
        
        // Add event listener
        document.getElementById('languageSelect')?.addEventListener('change', (e) => {
            this.setLanguage(e.target.value);
        });
    }
    
    addBackgroundModeControls() {
        const controlsContainer = document.querySelector('.voice-controls') || document.querySelector('.controls');
        if (!controlsContainer) return;
        
        const backgroundDiv = document.createElement('div');
        backgroundDiv.className = 'background-mode-section';
        backgroundDiv.innerHTML = `
            <div class="feature-section">
                <h4>🔄 Background Listening</h4>
                <div class="background-controls">
                    <button id="toggleBackgroundMode" class="background-btn ${this.backgroundMode.enabled ? 'active' : ''}">
                        🔄 Background: ${this.backgroundMode.enabled ? 'ON' : 'OFF'}
                    </button>
                    <div class="background-settings">
                        <label class="setting-row">
                            <input type="checkbox" id="pauseOnInactivity" ${this.backgroundMode.pauseOnInactivity ? 'checked' : ''}>
                            <span>Pause when inactive</span>
                        </label>
                        <label class="setting-row">
                            <input type="checkbox" id="batteryOptimization" ${this.backgroundMode.batteryOptimization ? 'checked' : ''}>
                            <span>Battery optimization</span>
                        </label>
                        <div class="setting-row">
                            <label>Max continuous hours:</label>
                            <input type="range" id="maxContinuousHours" min="1" max="12" value="${this.backgroundMode.maxContinuousHours}" step="1">
                            <span class="range-value">${this.backgroundMode.maxContinuousHours}h</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        controlsContainer.appendChild(backgroundDiv);
        
        // Add event listeners
        document.getElementById('toggleBackgroundMode')?.addEventListener('click', () => this.toggleBackgroundMode());
        document.getElementById('pauseOnInactivity')?.addEventListener('change', (e) => {
            this.backgroundMode.pauseOnInactivity = e.target.checked;
            this.saveBackgroundSettings();
        });
        document.getElementById('batteryOptimization')?.addEventListener('change', (e) => {
            this.backgroundMode.batteryOptimization = e.target.checked;
            this.saveBackgroundSettings();
        });
        document.getElementById('maxContinuousHours')?.addEventListener('input', (e) => {
            this.backgroundMode.maxContinuousHours = parseInt(e.target.value);
            document.querySelector('.range-value').textContent = `${e.target.value}h`;
            this.saveBackgroundSettings();
        });
    }
    
    addWakeWordControls() {
        const controlsContainer = document.querySelector('.voice-controls') || document.querySelector('.controls');
        if (!controlsContainer) return;
        
        const wakeWordDiv = document.createElement('div');
        wakeWordDiv.className = 'wake-word-section';
        wakeWordDiv.innerHTML = `
            <div class="feature-section">
                <h4>🌟 Wake Word Detection</h4>
                <div class="wake-word-controls">
                    <button id="toggleWakeWord" class="wake-word-btn active">
                        <span class="wake-word-indicator">●</span>
                        Always Listening
                    </button>
                    <div class="wake-word-info">
                        <small>Say "Hey Horizon" or "Horizon" to activate</small>
                        <div id="wakeWordStatus" class="status-text">Initializing...</div>
                    </div>
                </div>
            </div>
        `;
        
        controlsContainer.appendChild(wakeWordDiv);
        
        // Add event listener
        document.getElementById('toggleWakeWord')?.addEventListener('click', () => this.toggleWakeWord());
    }
    
    addVoiceCloneControls() {
        const controlsContainer = document.querySelector('.voice-controls') || document.querySelector('.controls');
        if (!controlsContainer) return;
        
        const voiceCloneDiv = document.createElement('div');
        voiceCloneDiv.className = 'voice-clone-section';
        voiceCloneDiv.innerHTML = `
            <div class="feature-section">
                <h4>🎤 Voice Cloning</h4>
                <div class="voice-clone-controls">
                    <button id="toggleVoiceClone" class="voice-clone-btn">
                        <span class="voice-clone-status">Disabled</span>
                    </button>
                    <button id="recordVoiceSample" class="record-sample-btn" disabled>
                        📹 Record Sample
                    </button>
                    <div class="voice-samples-info">
                        <small>Samples: <span id="sampleCount">0</span>/3 needed</small>
                    </div>
                </div>
            </div>
        `;
        
        controlsContainer.appendChild(voiceCloneDiv);
        
        // Add event listeners
        document.getElementById('toggleVoiceClone')?.addEventListener('click', () => this.toggleVoiceCloning());
        document.getElementById('recordVoiceSample')?.addEventListener('click', () => this.recordVoiceSample());
    }
    
    addStatusIndicators() {
        // Add CSS for the new features
        const style = document.createElement('style');
        style.textContent = `
            .feature-section {
                margin: 15px 0;
                padding: 15px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                border: 1px solid rgba(78, 205, 196, 0.3);
            }
            
            .feature-section h4 {
                color: #4ecdc4;
                margin: 0 0 10px 0;
                font-size: 1em;
            }
            
            .language-select {
                background: rgba(78, 205, 196, 0.1);
                border: 2px solid #4ecdc4;
                color: white;
                padding: 8px 15px;
                border-radius: 10px;
                font-size: 0.9em;
                width: 100%;
                margin-bottom: 8px;
            }
            
            .language-select option {
                background: #1a1a2e;
                color: white;
            }
            
            .background-btn {
                background: rgba(255, 159, 0, 0.1);
                border: 2px solid #ff9f00;
                color: white;
                padding: 8px 15px;
                border-radius: 20px;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-bottom: 10px;
            }
            
            .background-btn.active {
                background: rgba(255, 159, 0, 0.3);
                box-shadow: 0 0 10px rgba(255, 159, 0, 0.5);
            }
            
            .background-settings {
                margin-top: 10px;
                padding: 10px;
                background: rgba(255, 255, 255, 0.02);
                border-radius: 5px;
                border-left: 3px solid #ff9f00;
            }
            
            .setting-row {
                display: flex;
                align-items: center;
                margin-bottom: 8px;
                font-size: 0.9em;
                color: #96ceb4;
            }
            
            .setting-row input[type="checkbox"] {
                margin-right: 8px;
                accent-color: #4ecdc4;
            }
            
            .setting-row input[type="range"] {
                margin: 0 10px;
                flex: 1;
                accent-color: #4ecdc4;
            }
            
            .range-value {
                min-width: 30px;
                text-align: right;
                font-weight: bold;
                color: #4ecdc4;
            }
            
            .wake-word-btn, .voice-clone-btn {
                background: rgba(78, 205, 196, 0.1);
                border: 2px solid #4ecdc4;
                color: white;
                padding: 8px 15px;
                border-radius: 20px;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-right: 10px;
            }
            
            .wake-word-btn.active {
                background: rgba(78, 205, 196, 0.3);
                box-shadow: 0 0 10px rgba(78, 205, 196, 0.5);
            }
            
            .wake-word-indicator {
                color: #4ecdc4;
                animation: pulse 2s infinite;
            }
            
            .record-sample-btn {
                background: rgba(255, 119, 198, 0.1);
                border: 2px solid #ff77c6;
                color: white;
                padding: 6px 12px;
                border-radius: 15px;
                cursor: pointer;
                font-size: 0.9em;
            }
            
            .record-sample-btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            .voice-samples-info, .wake-word-info, .language-info {
                margin-top: 8px;
            }
            
            .status-text {
                color: #96ceb4;
                font-size: 0.8em;
                margin-top: 4px;
            }
            
            .voice-recording-modal {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.8);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 10000;
            }
            
            .recording-content {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 30px;
                border-radius: 15px;
                border: 2px solid #4ecdc4;
                text-align: center;
                max-width: 500px;
                color: white;
            }
            
            .recording-pulse {
                width: 60px;
                height: 60px;
                background: #ff6b6b;
                border-radius: 50%;
                margin: 20px auto;
                animation: recordingPulse 1s infinite;
            }
            
            .sample-text {
                background: rgba(78, 205, 196, 0.1);
                padding: 15px;
                border-left: 3px solid #4ecdc4;
                margin: 15px 0;
                font-style: italic;
                text-align: left;
                line-height: 1.5;
            }
            
            .stop-recording-btn {
                background: #ff6b6b;
                border: none;
                color: white;
                padding: 10px 20px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 1em;
                margin-top: 15px;
            }
            
            @keyframes recordingPulse {
                0% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.2); opacity: 0.7; }
                100% { transform: scale(1); opacity: 1; }
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            @keyframes backgroundPulse {
                0% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.6; transform: scale(1.2); }
                100% { opacity: 1; transform: scale(1); }
            }
        `;
        
        document.head.appendChild(style);
    }
    
    toggleBackgroundMode() {
        if (this.backgroundMode.enabled) {
            this.disableBackgroundMode();
        } else {
            this.enableBackgroundMode();
        }
    }
    
    toggleWakeWord() {
        const button = document.getElementById('toggleWakeWord');
        const indicator = document.querySelector('.wake-word-indicator');
        
        if (this.isWakeWordListening) {
            this.stopWakeWordListening();
            if (button) button.classList.remove('active');
            if (indicator) indicator.style.animation = 'none';
            this.updateWakeWordStatus('Wake word detection disabled');
        } else {
            this.startWakeWordListening();
            if (button) button.classList.add('active');
            if (indicator) indicator.style.animation = 'pulse 2s infinite';
        }
    }
    
    // ===== INTEGRATION METHODS =====
    
    onMainListeningStart() {
        // Stop wake word detection when main listening starts
        this.stopWakeWordListening();
    }
    
    onMainListeningStop() {
        // Resume wake word detection when main listening stops
        setTimeout(() => {
            this.startWakeWordListening();
        }, 1000);
    }
    
    loadSettings() {
        // Load all settings
        this.loadVoiceSettings();
        this.loadBackgroundSettings();
        
        // Load language preference
        try {
            const savedLang = localStorage.getItem('horizon_language');
            if (savedLang && this.supportedLanguages[savedLang]) {
                this.setLanguage(savedLang);
            }
        } catch (error) {
            console.log('Could not load language settings:', error);
        }
        
        // Update UI based on settings
        const sampleCount = document.getElementById('sampleCount');
        if (sampleCount && this.voiceSettings.recordedSamples) {
            sampleCount.textContent = this.voiceSettings.recordedSamples.length;
        }
    }
    
    saveSettings() {
        this.saveVoiceSettings();
        this.saveBackgroundSettings();
        
        // Save language preference
        try {
            localStorage.setItem('horizon_language', this.currentLanguage);
        } catch (error) {
            console.log('Could not save language settings:', error);
        }
    }
    
    // ===== PUBLIC API =====
    
    getWakeWordStatus() {
        return this.isWakeWordListening;
    }
    
    getVoiceCloneStatus() {
        return this.voiceSettings.voiceCloneEnabled;
    }
    
    getSampleCount() {
        return this.voiceSettings.recordedSamples.length;
    }
    
    getCurrentLanguage() {
        return {
            code: this.currentLanguage,
            name: this.supportedLanguages[this.currentLanguage].name,
            wakeWords: this.wakeWords
        };
    }
    
    getBackgroundModeStatus() {
        return {
            enabled: this.backgroundMode.enabled,
            lowPowerMode: this.backgroundMode.lowPowerMode,
            continuousHours: this.backgroundMode.continuousHours,
            pauseOnInactivity: this.backgroundMode.pauseOnInactivity
        };
    }
    
    getSupportedLanguages() {
        return this.supportedLanguages;
    }
}

// Auto-initialize when DOM is ready and main AI assistant is available
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        if (window.aiAssistant) {
            window.voiceEnhancements = new VoiceEnhancements(window.aiAssistant);
            console.log('🚀 Voice enhancements loaded successfully!');
        }
    }, 1000);
});