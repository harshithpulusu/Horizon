// Enhanced AI Voice Assistant with advanced features

class EnhancedAIVoiceAssistant {
    constructor() {
        this.isListening = false;
        this.isSpeaking = false;
        this.recognition = null;
        this.wakeWordRecognition = null;  // New: separate recognition for wake words
        this.synthesis = window.speechSynthesis;
        this.currentPersonality = 'friendly';
        this.conversationHistory = [];
        this.activeTimers = [];
        this.activeReminders = [];
        this.lastResponse = '';
        this.lastInput = '';
        
        // Wake word detection properties
        this.isWakeWordMode = true;  // Always listening for wake words
        this.wakeWords = ['hey horizon', 'horizon', 'hey assistant', 'assistant'];
        this.wakeWordSensitivity = 0.7;
        this.wakeWordTimeout = null;
        this.isWakeWordListening = false;
        
        // Voice cloning properties
        this.voiceSettings = {
            enabled: false,
            userVoiceId: null,
            personalizedVoices: {},
            voiceCloneEnabled: false,
            recordedSamples: []
        };
        
        this.sessionId = null;
        this.conversationLength = 0;
        this.contextUsed = false;
        this.memorySystem = null;  // Will be set when memory system is available
        this.contextualIntelligence = null;  // Will be set when contextual system is available
        
        this.init();
        this.initEventListeners();
        this.loadActiveTimersReminders();
        this.startPeriodicUpdates();
        this.initializeSession();
        this.startWakeWordListening();  // Auto-start wake word detection
    }
    
    init() {
        // Get DOM elements first
        this.startBtn = document.getElementById('startListening');
        this.stopBtn = document.getElementById('stopListening');
        this.voiceInput = document.getElementById('voiceInput');
        this.sendBtn = document.getElementById('sendMessage');
        this.messagesContainer = document.getElementById('messages');
        this.personalitySelect = document.getElementById('personalitySelect');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.confidenceScore = document.getElementById('confidenceScore');
        this.conversationCount = document.getElementById('conversationCount');
        this.timersContainer = document.getElementById('activeTimers');
        this.remindersContainer = document.getElementById('activeReminders');
        this.feedbackContainer = document.getElementById('feedbackContainer');
        
        // Initialize speech recognition with better error handling
        this.initSpeechRecognition();
        this.initWakeWordDetection();  // Initialize wake word detection
        this.initVoiceCloning();       // Initialize voice cloning
        
        // Ensure personality selector is set to friendly by default
        if (this.personalitySelect) {
            this.personalitySelect.value = this.currentPersonality;
        }
        
        this.updateStatus('Ready - Say "Hey Horizon" to start');
    }
    
    initSpeechRecognition() {
        // Check if we're on HTTPS or localhost (required for speech recognition)
        const isSecure = location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
        
        if (!isSecure) {
            this.addMessage('System', 'Speech recognition requires HTTPS or localhost. Voice features disabled.', 'error');
            this.disableVoiceFeatures();
            return;
        }
        
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            try {
                const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
                this.recognition = new SpeechRecognition();
                this.recognition.continuous = false;
                this.recognition.interimResults = false;
                this.recognition.lang = 'en-US';
                this.recognition.maxAlternatives = 1;
                
                this.recognition.onstart = () => {
                    console.log('Speech recognition started');
                    this.updateStatus('Listening... Speak now!');
                };
                
                this.recognition.onresult = (event) => this.handleSpeechResult(event);
                this.recognition.onerror = (event) => this.handleSpeechError(event);
                this.recognition.onend = () => this.handleSpeechEnd();
                
                this.addMessage('System', 'Speech recognition initialized successfully! 🎤', 'system');
            } catch (error) {
                console.error('Error initializing speech recognition:', error);
                this.addMessage('System', 'Error initializing speech recognition: ' + error.message, 'error');
                this.disableVoiceFeatures();
            }
        } else {
            this.addMessage('System', 'Speech recognition not supported in this browser. Try Chrome or Safari.', 'error');
            this.disableVoiceFeatures();
        }
    }
    
    disableVoiceFeatures() {
        if (this.startBtn) {
            this.startBtn.disabled = true;
            this.startBtn.textContent = '🎤 Voice Not Available';
        }
        if (this.stopBtn) {
            this.stopBtn.disabled = true;
        }
    }
    
    initWakeWordDetection() {
        // Only initialize if speech recognition is available
        if (!this.recognition) {
            return;
        }
        
        try {
            const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
            this.wakeWordRecognition = new SpeechRecognition();
            this.wakeWordRecognition.continuous = true;  // Keep listening
            this.wakeWordRecognition.interimResults = true;  // Get partial results
            this.wakeWordRecognition.lang = 'en-US';
            this.wakeWordRecognition.maxAlternatives = 3;
            
            this.wakeWordRecognition.onresult = (event) => this.handleWakeWordResult(event);
            this.wakeWordRecognition.onerror = (event) => {
                console.log('Wake word detection error:', event.error);
                // Restart wake word detection on error (unless it's aborted)
                if (event.error !== 'aborted' && this.isWakeWordListening) {
                    setTimeout(() => this.startWakeWordListening(), 1000);
                }
            };
            
            this.wakeWordRecognition.onend = () => {
                // Restart wake word detection when it ends
                if (this.isWakeWordListening && !this.isListening) {
                    setTimeout(() => this.startWakeWordListening(), 100);
                }
            };
            
            console.log('Wake word detection initialized successfully');
        } catch (error) {
            console.error('Error initializing wake word detection:', error);
        }
    }
    
    startWakeWordListening() {
        if (!this.wakeWordRecognition || this.isWakeWordListening || this.isListening) {
            return;
        }
        
        try {
            this.isWakeWordListening = true;
            this.wakeWordRecognition.start();
            console.log('Wake word listening started...');
            this.updateStatus('Listening for "Hey Horizon"...');
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
        
        if (wakeWordDetected && !this.isListening) {
            console.log('Wake word detected!');
            this.onWakeWordDetected(transcript);
        }
    }
    
    onWakeWordDetected(transcript) {
        // Stop wake word listening temporarily
        this.stopWakeWordListening();
        
        // Visual feedback
        this.showWakeWordDetectedFeedback();
        
        // Audio feedback (quick beep)
        this.playWakeWordSound();
        
        // Start main speech recognition for the command
        setTimeout(() => {
            this.startListening();
        }, 300);
        
        // Resume wake word listening after a timeout
        this.wakeWordTimeout = setTimeout(() => {
            if (!this.isListening) {
                this.startWakeWordListening();
            }
        }, 5000);
    }
    
    showWakeWordDetectedFeedback() {
        // Update status
        this.updateStatus('Wake word detected! Listening for command...');
        
        // Visual pulse effect
        const statusElement = document.getElementById('statusIndicator');
        if (statusElement) {
            statusElement.style.animation = 'pulse 0.5s ease-in-out';
            setTimeout(() => {
                if (statusElement.style.animation) {
                    statusElement.style.animation = '';
                }
            }, 500);
        }
        
        // Show toast notification
        if (window.professionalUI) {
            window.professionalUI.showToast('Wake word detected! 👂', 'success', 2000);
        }
    }
    
    playWakeWordSound() {
        // Create a simple beep sound using Web Audio API
        if (typeof AudioContext !== 'undefined' || typeof webkitAudioContext !== 'undefined') {
            try {
                const audioContext = new (AudioContext || webkitAudioContext)();
                const oscillator = audioContext.createOscillator();
                const gainNode = audioContext.createGain();
                
                oscillator.connect(gainNode);
                gainNode.connect(audioContext.destination);
                
                oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
                oscillator.frequency.exponentialRampToValueAtTime(400, audioContext.currentTime + 0.1);
                
                gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
                
                oscillator.start(audioContext.currentTime);
                oscillator.stop(audioContext.currentTime + 0.1);
            } catch (error) {
                console.log('Could not play wake word sound:', error);
            }
        }
    }
    
    // Initialize voice cloning features
    initVoiceCloning() {
        this.loadVoiceSettings();
        this.addVoiceCloneToggle();
        this.initializeElevenLabsVoices();
    }
    
    addVoiceCloneToggle() {
        // Add voice cloning toggle to the UI
        const controlsContainer = document.querySelector('.voice-controls') || document.querySelector('.controls');
        if (!controlsContainer) return;
        
        const voiceCloneDiv = document.createElement('div');
        voiceCloneDiv.className = 'voice-clone-controls';
        voiceCloneDiv.innerHTML = `
            <div class="voice-clone-section">
                <h4>🎙️ Voice Cloning</h4>
                <button id="toggleVoiceClone" class="voice-clone-btn">
                    <span class="voice-clone-status">Disabled</span>
                </button>
                <button id="recordVoiceSample" class="record-sample-btn" disabled>
                    📹 Record Voice Sample
                </button>
                <div class="voice-samples-info">
                    <small>Samples recorded: <span id="sampleCount">0</span>/3</small>
                </div>
            </div>
        `;
        
        controlsContainer.appendChild(voiceCloneDiv);
        
        // Add event listeners
        document.getElementById('toggleVoiceClone')?.addEventListener('click', () => this.toggleVoiceCloning());
        document.getElementById('recordVoiceSample')?.addEventListener('click', () => this.recordVoiceSample());
    }

    addWakeWordUI() {
        // Add wake word toggle to the UI
        const controlsDiv = document.querySelector('.controls');
        if (controlsDiv && !document.getElementById('wakeWordToggle')) {
            const wakeWordDiv = document.createElement('div');
            wakeWordDiv.className = 'wake-word-controls';
            wakeWordDiv.innerHTML = `
                <label class="wake-word-label">
                    <input type="checkbox" id="wakeWordToggle" class="wake-word-checkbox">
                    <span class="wake-word-text">🌟 Always Listening Mode</span>
                    <small>Say "Hey Horizon" or "Horizon" to activate</small>
                </label>
            `;
            
            // Add some CSS for the wake word toggle
            const style = document.createElement('style');
            style.textContent = `
                .wake-word-controls {
                    margin: 10px 0;
                    padding: 15px;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 10px;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }
                .wake-word-label {
                    display: flex;
                    flex-direction: column;
                    gap: 5px;
                    cursor: pointer;
                    color: white;
                }
                .wake-word-text {
                    font-weight: 500;
                    font-size: 14px;
                }
                .wake-word-label small {
                    color: rgba(255, 255, 255, 0.7);
                    font-size: 12px;
                }
                .wake-word-checkbox {
                    margin-right: 8px;
                    transform: scale(1.2);
                }
            `;
            document.head.appendChild(style);
            
            controlsDiv.appendChild(wakeWordDiv);
            
            // Add event listener for the toggle
            const toggle = document.getElementById('wakeWordToggle');
            toggle.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.startWakeWordDetection();
                } else {
                    this.stopWakeWordDetection();
                }
            });
        }
    }
    
    startWakeWordDetection() {
        if (!this.wakeWordRecognition || this.isWakeWordMode) {
            return;
        }
        
        try {
            this.isWakeWordMode = true;
            this.wakeWordRecognition.start();
            this.updateStatus('👂 Listening for "Hey Horizon"...');
            console.log('Wake word detection started');
        } catch (error) {
            console.error('Error starting wake word detection:', error);
            this.isWakeWordMode = false;
        }
    }
    
    stopWakeWordDetection() {
        if (!this.wakeWordRecognition || !this.isWakeWordMode) {
            return;
        }
        
        try {
            this.isWakeWordMode = false;
            this.wakeWordRecognition.abort();
            this.updateStatus('Ready');
            console.log('Wake word detection stopped');
        } catch (error) {
            console.error('Error stopping wake word detection:', error);
        }
    }
    
    handleWakeWordResult(event) {
        const results = Array.from(event.results);
        const lastResult = results[results.length - 1];
        
        if (lastResult && lastResult[0]) {
            const transcript = lastResult[0].transcript.toLowerCase().trim();
            const confidence = lastResult[0].confidence || 0;
            
            console.log('Wake word transcript:', transcript, 'Confidence:', confidence);
            
            // Check if any wake word was detected
            const wakeWordDetected = this.wakeWords.some(wakeWord => {
                return transcript.includes(wakeWord.toLowerCase()) && 
                       confidence >= this.wakeWordSensitivity;
            });
            
            if (wakeWordDetected && !this.isListening) {
                console.log('Wake word detected!');
                this.handleWakeWordDetected(transcript);
            }
        }
    }
    
    handleWakeWordDetected(transcript) {
        // Stop wake word detection temporarily
        this.wakeWordRecognition.abort();
        
        // Visual/audio feedback
        this.updateStatus('🌟 Wake word detected! Listening...');
        this.addMessage('System', '🌟 Wake word detected! How can I help?', 'system');
        
        // Start regular listening for the actual command
        setTimeout(() => {
            this.startListening();
        }, 500);
        
        // Restart wake word detection after the command is processed
        setTimeout(() => {
            if (this.isWakeWordMode && !this.isListening) {
                this.startWakeWordDetection();
            }
        }, 10000); // Wait 10 seconds before resuming wake word detection
    }
    
    initEventListeners() {
        // Voice controls
        this.startBtn?.addEventListener('click', () => this.startListening());
        this.stopBtn?.addEventListener('click', () => this.stopListening());
        this.sendBtn?.addEventListener('click', () => this.sendMessage());
        
        // Microphone test
        const testMicBtn = document.getElementById('testMicBtn');
        testMicBtn?.addEventListener('click', () => this.testMicrophone());
        
        // Clear conversation button
        const clearConversationBtn = document.getElementById('clearConversationBtn');
        clearConversationBtn?.addEventListener('click', () => this.clearConversation());
        
        // Text input
        this.voiceInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });
        
        // Personality selection
        this.personalitySelect?.addEventListener('change', (e) => {
            this.setPersonality(e.target.value);
        });
        
        // Add quick command listeners directly to existing buttons
        this.initQuickCommandListeners();
        
        // Feedback buttons and other dynamic elements
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('feedback-btn')) {
                this.submitFeedback(e.target.dataset.feedback);
            } else if (e.target.classList.contains('cancel-timer-btn')) {
                const timerId = parseInt(e.target.dataset.timerId);
                this.cancelTimer(timerId);
            } else if (e.target.classList.contains('cancel-reminder-btn')) {
                const reminderId = parseInt(e.target.dataset.reminderId);
                this.cancelReminder(reminderId);
            }
        });
    }
    
    initQuickCommandListeners() {
        // Add listeners to quick command buttons
        const quickCmdButtons = document.querySelectorAll('.quick-cmd-btn');
        console.log('Found quick command buttons:', quickCmdButtons.length);
        
        if (quickCmdButtons.length === 0) {
            console.warn('No quick command buttons found!');
            return;
        }
        
        quickCmdButtons.forEach((button, index) => {
            console.log(`Button ${index}:`, button.dataset.command);
            button.addEventListener('click', (e) => {
                e.preventDefault();
                console.log('Quick command button clicked:', button.dataset.command);
                this.handleQuickCommand(button.dataset.command, button);
            });
        });
    }
    
    async startListening() {
        if (!this.recognition) {
            this.addMessage('AI', '❌ Speech recognition not available. Please check browser permissions or try a different browser.', 'error');
            return;
        }
        
        // Check if already listening
        if (this.isListening) {
            this.addMessage('System', 'Already listening...', 'system');
            return;
        }
        
        try {
            // Request microphone permission explicitly
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                try {
                    await navigator.mediaDevices.getUserMedia({ audio: true });
                    console.log('Microphone permission granted');
                } catch (permissionError) {
                    this.addMessage('AI', '❌ Microphone permission denied. Please allow microphone access and try again.', 'error');
                    return;
                }
            }
            
            this.isListening = true;
            this.updateStatus('Starting...');
            
            // Update UI
            if (this.startBtn) {
                this.startBtn.disabled = true;
                this.startBtn.textContent = '🎤 Listening...';
            }
            if (this.stopBtn) {
                this.stopBtn.disabled = false;
            }
            
            // Add visual feedback
            this.statusIndicator?.classList.add('listening');
            
            // Start recognition
            this.recognition.start();
            
            this.addMessage('System', '🎤 Listening started! Speak now...', 'system');
            
        } catch (error) {
            console.error('Speech recognition error:', error);
            this.addMessage('AI', `❌ Error starting speech recognition: ${error.message}`, 'error');
            this.resetListeningState();
        }
    }
    
    stopListening() {
        if (this.recognition && this.isListening) {
            try {
                this.recognition.stop();
                this.addMessage('System', '⏹️ Stopping speech recognition...', 'system');
            } catch (error) {
                console.error('Error stopping recognition:', error);
            }
        } else {
            this.addMessage('System', 'Not currently listening.', 'system');
        }
        this.resetListeningState();
    }
    
    resetListeningState() {
        this.isListening = false;
        this.updateStatus('Ready');
        
        // Reset buttons
        if (this.startBtn) {
            this.startBtn.disabled = false;
            this.startBtn.textContent = '🎤 Start Listening';
        }
        if (this.stopBtn) {
            this.stopBtn.disabled = true;
        }
        
        // Remove visual feedback
        this.statusIndicator?.classList.remove('listening');
    }
    
    handleSpeechResult(event) {
        console.log('Speech result received:', event);
        
        if (event.results && event.results.length > 0) {
            const transcript = event.results[0][0].transcript;
            const confidence = event.results[0][0].confidence;
            
            console.log('Transcript:', transcript, 'Confidence:', confidence);
            
            this.voiceInput.value = transcript;
            this.addMessage('You', `🎤 "${transcript}" (${Math.round(confidence * 100)}% confidence)`, 'user');
            this.processInput(transcript);
        } else {
            this.addMessage('System', '❌ No speech detected. Please try again.', 'error');
        }
        
        this.resetListeningState();
    }
    
    handleSpeechError(event) {
        console.error('Speech recognition error:', event);
        
        let errorMessage = 'Speech recognition error: ';
        switch(event.error) {
            case 'no-speech':
                errorMessage += 'No speech detected. Please try again.';
                break;
            case 'audio-capture':
                errorMessage += 'Microphone not available. Please check your microphone settings.';
                break;
            case 'not-allowed':
                errorMessage += 'Microphone permission denied. Please allow microphone access.';
                break;
            case 'network':
                errorMessage += 'Network error. Please check your internet connection.';
                break;
            case 'service-not-allowed':
                errorMessage += 'Speech service not allowed. Try using HTTPS.';
                break;
            default:
                errorMessage += event.error;
        }
        
        this.addMessage('AI', `❌ ${errorMessage}`, 'error');
        this.updateStatus(`Error: ${event.error}`);
        this.resetListeningState();
    }
    
    handleSpeechEnd() {
        console.log('Speech recognition ended');
        this.resetListeningState();
        
        // Restart wake word detection if it was enabled
        if (this.isWakeWordMode && !this.isListening) {
            setTimeout(() => {
                this.startWakeWordDetection();
            }, 1000); // Wait 1 second before restarting wake word detection
        }
    }
    
    async sendMessage() {
        const input = this.voiceInput.value.trim();
        if (!input) return;
        
        this.addMessage('You', input, 'user');
        this.voiceInput.value = '';
        await this.processInput(input);
    }
    
    handleQuickCommand(command, buttonElement) {
        // Handle quick command button clicks
        console.log('Quick command clicked:', command);
        
        // Add visual feedback to the clicked button
        if (buttonElement) {
            buttonElement.style.background = 'rgba(69, 183, 209, 0.3)';
            buttonElement.style.transform = 'translateX(8px) scale(0.95)';
            
            setTimeout(() => {
                buttonElement.style.background = '';
                buttonElement.style.transform = '';
            }, 200);
        }
        
        // Add the command to the chat and display it
        this.addMessage('You', command, 'user');
        
        // Process the command
        this.processInput(command);
        
        // Update status
        this.updateStatus('Processing quick command...');
    }
    
    async processInput(input) {
        this.updateStatus('Processing...');
        this.lastInput = input;
        
        // Enhance input with contextual intelligence
        let enhancedInput = input;
        if (this.contextualIntelligence) {
            enhancedInput = this.contextualIntelligence.enhanceUserMessage(input);
        }
        
        // Record user interaction patterns for learning
        if (this.memorySystem) {
            this.memorySystem.recordInteraction('user_message', {
                message: input,
                enhanced_message: enhancedInput,
                timestamp: new Date().toISOString(),
                message_length: input.length,
                session_id: this.sessionId
            });
        }
        
        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    input: enhancedInput,
                    original_input: input,
                    personality: this.currentPersonality,
                    session_id: this.sessionId,
                    context_data: this.getContextualData()
                })
            });
            
            if (!response.ok) throw new Error('Network response was not ok');
            
            const data = await response.json();
            
            // Update session information from response
            if (data.session_id) {
                this.sessionId = data.session_id;
            }
            if (data.conversation_length !== undefined) {
                this.conversationLength = data.conversation_length;
            }
            if (data.context_used !== undefined) {
                this.contextUsed = data.context_used;
            }
            
            this.handleAIResponse(data);
            
            // Learn from the interaction
            if (this.memorySystem && data.response) {
                this.learnFromInteraction(input, data.response);
            }
            
        } catch (error) {
            console.error('Error processing input:', error);
            this.addMessage('AI', 'Sorry, I encountered an error processing your request.', 'error');
            this.updateStatus('Error');
        }
    }
    
    // Learn from user-AI interactions
    learnFromInteraction(userInput, aiResponse) {
        // Extract important facts from user input
        const personalKeywords = ['my name is', 'i am', 'i work', 'i live', 'i like', 'i prefer', 'remember'];
        const lowerInput = userInput.toLowerCase();
        
        personalKeywords.forEach(keyword => {
            if (lowerInput.includes(keyword)) {
                this.recordConversationFact(userInput, 0.8);
                
                // Store personal context
                this.updatePersonalContext({
                    key: `user_statement_${Date.now()}`,
                    value: userInput,
                    category: 'personal_info',
                    confidence: 0.8
                });
            }
        });
        
        // Learn response preferences
        if (this.memorySystem) {
            this.memorySystem.learnUserPreference({
                preference_category: 'response_style',
                preference_name: 'interaction_pattern',
                preference_value: {
                    user_input_style: this.analyzeInputStyle(userInput),
                    preferred_response_length: aiResponse.length > 200 ? 'detailed' : aiResponse.length > 50 ? 'medium' : 'brief',
                    interaction_timestamp: new Date().toISOString()
                },
                learning_source: 'conversation_analysis',
                confidence_level: 0.6
            });
        }
    }
    
    analyzeInputStyle(input) {
        if (input.length < 20) return 'brief';
        if (input.includes('?')) return 'questioning';
        if (input.includes('please') || input.includes('thank')) return 'polite';
        if (input.includes('!')) return 'enthusiastic';
        return 'neutral';
    }
    
    handleAIResponse(data) {
        this.lastResponse = data.response;
        this.addMessage('AI', data.response, 'ai', data);
        // Text-to-speech disabled
        // this.speak(data.response);
        
        // Update statistics with realistic values
        const confidence = Math.round((data.confidence || 0.82) * 100);
        this.confidenceScore.textContent = confidence + '%';
        
        // Update conversation count with session information
        const conversationCount = data.conversation_length || (parseInt(this.conversationCount.textContent) + 1);
        this.conversationCount.textContent = conversationCount;
        
        // Show context indicator if context was used
        if (data.context_used && data.has_context) {
            this.showContextIndicator();
        }
        
        // Update professional UI metrics if available
        if (window.professionalUI) {
            const responseTime = parseFloat(data.response_time) || 0.8;
            window.professionalUI.updateMetrics(responseTime, data.confidence || 0.82);
        }
        
        // Show feedback options
        this.showFeedbackOptions();
        
        // Update timers and reminders if needed
        if (data.intent === 'timer' || data.intent === 'reminder') {
            setTimeout(() => this.loadActiveTimersReminders(), 1000);
        }
        
        this.updateStatus('Ready');
    }
    
    showContextIndicator() {
        // Add a visual indicator that context was used
        const lastMessage = this.messagesContainer.lastElementChild;
        if (lastMessage) {
            const contextBadge = document.createElement('span');
            contextBadge.className = 'context-badge';
            contextBadge.innerHTML = '🧠 Context-aware';
            contextBadge.title = 'This response used previous conversation context';
            lastMessage.appendChild(contextBadge);
        }
    }
    
    formatMarkdownLinks(text) {
        // Convert markdown links [text](url) to HTML <a> tags
        return text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, linkText, url) => {
            // Add target="_blank" for external links
            const isExternal = url.startsWith('http://') || url.startsWith('https://');
            const target = isExternal ? ' target="_blank" rel="noopener noreferrer"' : '';
            
            // Add special styling for logo links
            const isLogoLink = linkText.toLowerCase().includes('logo') || linkText.toLowerCase().includes('view');
            const className = isLogoLink ? ' class="logo-link"' : '';
            
            return `<a href="${url}"${target}${className}>${linkText}</a>`;
        });
    }
    
    addMessage(sender, message, type, data = null, autoScroll = true) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const timestamp = new Date().toLocaleTimeString();
        
        // Check if message contains image URL (both external and local)
        const imageUrlPattern = /(https?:\/\/[^\s]+\.(jpg|jpeg|png|gif|webp|bmp)|\/static\/generated_images\/[^\s]+\.(jpg|jpeg|png|gif|webp|bmp))/gi;
        
        // Check for video URLs
        const videoUrlPattern = /(https?:\/\/[^\s]+\.(mp4|avi|mov|webm|mkv)|\/static\/videos\/[^\s]+\.(mp4|avi|mov|webm|mkv))/gi;
        
        // Check for GIF URLs (animated GIFs)
        const gifUrlPattern = /(https?:\/\/[^\s]+\.gif|\/static\/gifs\/[^\s]+\.gif)/gi;
        
        // Check for full localhost URLs pointing to images
        const fullLocalUrlPattern = /(http:\/\/127\.0\.0\.1:8080\/static\/generated_images\/[^\s]+\.(png|jpg|jpeg|gif|webp))/gi;
        
        // Check for full localhost URLs pointing to videos
        const fullLocalVideoPattern = /(http:\/\/127\.0\.0\.1:8080\/static\/videos\/[^\s]+\.(mp4|avi|mov|webm|mkv))/gi;
        
        // Check for full localhost URLs pointing to GIFs
        const fullLocalGifPattern = /(http:\/\/127\.0\.0\.1:8080\/static\/gifs\/[^\s]+\.gif)/gi;
        
        // Check for markdown image syntax: ![alt](url)
        const markdownImagePattern = /!\[.*?\]\((\/static\/generated_images\/[^)]+\.(png|jpg|jpeg|gif|webp))\)/gi;
        
        // Also check for video and gif paths that might be on their own line
        const localImagePattern = /\/static\/generated_images\/[\w\-\.]+\.(png|jpg|jpeg|gif|webp)/gi;
        const localVideoPattern = /\/static\/videos\/[\w\-\.]+\.(mp4|avi|mov|webm|mkv)/gi;
        const localGifPattern = /\/static\/gifs\/[\w\-\.]+\.gif/gi;
        
        // Even more specific pattern for the exact format we're generating
        const specificImagePattern = /\/static\/generated_images\/[a-f0-9\-]+\.png/gi;
        const specificVideoPattern = /\/static\/videos\/[a-f0-9\-]+\.mp4/gi;
        const specificGifPattern = /\/static\/gifs\/[a-f0-9\-]+\.gif/gi;
        
        let imageUrls = message.match(imageUrlPattern) || [];
        let videoUrls = message.match(videoUrlPattern) || [];
        let gifUrls = message.match(gifUrlPattern) || [];
        
        const fullLocalUrls = message.match(fullLocalUrlPattern) || [];
        const fullLocalVideos = message.match(fullLocalVideoPattern) || [];
        const fullLocalGifs = message.match(fullLocalGifPattern) || [];
        
        const localImages = message.match(localImagePattern) || [];
        const localVideos = message.match(localVideoPattern) || [];
        const localGifs = message.match(localGifPattern) || [];
        
        const specificImages = message.match(specificImagePattern) || [];
        const specificVideos = message.match(specificVideoPattern) || [];
        const specificGifs = message.match(specificGifPattern) || [];
        
        // Extract URLs from markdown image syntax
        let markdownMatch;
        const markdownImages = [];
        while ((markdownMatch = markdownImagePattern.exec(message)) !== null) {
            markdownImages.push(markdownMatch[1]);
        }
        
        // Combine all patterns and remove duplicates
        imageUrls = [...imageUrls, ...fullLocalUrls, ...localImages, ...specificImages, ...markdownImages].filter((url, index, self) => self.indexOf(url) === index);
        videoUrls = [...videoUrls, ...fullLocalVideos, ...localVideos, ...specificVideos].filter((url, index, self) => self.indexOf(url) === index);
        gifUrls = [...gifUrls, ...fullLocalGifs, ...localGifs, ...specificGifs].filter((url, index, self) => self.indexOf(url) === index);
        
        // Combine all media URLs for message cleaning
        const allMediaUrls = [...imageUrls, ...videoUrls, ...gifUrls];
        
        // Remove image URLs from the message text if they will be displayed as images
        let displayMessage = message;
        if (allMediaUrls && allMediaUrls.length > 0) {
            // Remove markdown image syntax
            displayMessage = displayMessage.replace(/!\[.*?\]\([^)]+\)/gi, '');
            
            // Remove plain media URLs
            allMediaUrls.forEach(url => {
                displayMessage = displayMessage.replace(url, '').trim();
            });
            
            // Clean up any extra line breaks
            displayMessage = displayMessage.replace(/\n\s*\n/g, '\n').trim();
        }
        
        // Convert markdown links to HTML (for clickable logo URLs)
        displayMessage = this.formatMarkdownLinks(displayMessage);
        
        // Convert bold markdown to HTML
        displayMessage = displayMessage.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        let messageContent = `
            <div class="message-header">
                <strong>${sender}</strong>
                <span class="timestamp">${timestamp}</span>
            </div>
            <div class="message-content">${displayMessage}</div>
        `;
        
        // Add image display if image URLs are found
        if (imageUrls && imageUrls.length > 0) {
            messageContent += '<div class="image-gallery">';
            imageUrls.forEach(url => {
                messageContent += `
                    <div class="generated-image-container">
                        <img src="${url}" 
                             alt="Generated AI Image" 
                             class="generated-image"
                             onclick="this.classList.toggle('fullscreen')"
                             onerror="this.style.display='none'; this.nextElementSibling.style.display='block';"
                             onload="professionalUI.showToast('🎨 Image loaded successfully!', 'success', 3000);">
                        <div class="image-error" style="display: none;">
                            <p>❌ Failed to load image</p>
                            <a href="${url}" target="_blank">View original link</a>
                        </div>
                        <div class="image-actions">
                            <button onclick="window.open('${url}', '_blank')" class="action-btn">
                                🔗 Open in New Tab
                            </button>
                            <button onclick="navigator.clipboard.writeText('${url}')" class="action-btn">
                                📋 Copy URL
                            </button>
                        </div>
                    </div>
                `;
            });
            messageContent += '</div>';
        }

        // Add video display if video URLs are found
        if (videoUrls && videoUrls.length > 0) {
            messageContent += '<div class="video-gallery">';
            videoUrls.forEach(url => {
                messageContent += `
                    <div class="generated-video-container">
                        <video controls class="generated-video"
                               onloadstart="professionalUI.showToast('🎥 Video loading...', 'info', 2000);"
                               oncanplay="professionalUI.showToast('🎥 Video ready to play!', 'success', 3000);"
                               onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                            <source src="${url}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                        <div class="video-error" style="display: none;">
                            <p>❌ Failed to load video</p>
                            <a href="${url}" target="_blank">View original link</a>
                        </div>
                        <div class="video-actions">
                            <button onclick="window.open('${url}', '_blank')" class="action-btn">
                                🔗 Open in New Tab
                            </button>
                            <button onclick="navigator.clipboard.writeText('${url}')" class="action-btn">
                                📋 Copy URL
                            </button>
                        </div>
                    </div>
                `;
            });
            messageContent += '</div>';
        }

        // Add GIF display if GIF URLs are found
        if (gifUrls && gifUrls.length > 0) {
            messageContent += '<div class="gif-gallery">';
            gifUrls.forEach(url => {
                messageContent += `
                    <div class="generated-gif-container">
                        <img src="${url}" 
                             alt="Generated AI GIF" 
                             class="generated-gif"
                             onclick="this.classList.toggle('fullscreen')"
                             onload="professionalUI.showToast('🎞️ GIF loaded successfully!', 'success', 3000);"
                             onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                        <div class="gif-error" style="display: none;">
                            <p>❌ Failed to load GIF</p>
                            <a href="${url}" target="_blank">View original link</a>
                        </div>
                        <div class="gif-actions">
                            <button onclick="window.open('${url}', '_blank')" class="action-btn">
                                🔗 Open in New Tab
                            </button>
                            <button onclick="navigator.clipboard.writeText('${url}')" class="action-btn">
                                📋 Copy URL
                            </button>
                        </div>
                    </div>
                `;
            });
            messageContent += '</div>';
        }
        
        // Add enhanced metadata for AI responses
        if (data && type === 'ai') {
            const confidence = data.confidence || 0.82;
            const aiSource = data.ai_source || 'fallback';
            const responseTime = data.response_time || '0.8s';
            const intent = data.intent || 'general';
            const contextUsed = data.context_used ? ' | 🧠 Context-aware' : '';
            
            messageContent += `
                <div class="message-metadata">
                    <small>
                        Response Time: ${responseTime} | 
                        Confidence: ${Math.round(confidence * 100)}% |
                        Source: ${aiSource === 'chatgpt' ? 'ChatGPT API' : 'Smart Fallback'} |
                        Intent: ${intent}${contextUsed}
                    </small>
                </div>
            `;
        }
        
        messageDiv.innerHTML = messageContent;
        this.messagesContainer.appendChild(messageDiv);
        
        // Only auto-scroll for new messages, not historical ones
        if (autoScroll) {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }
        
        // Store in conversation history
        this.conversationHistory.push({
            sender, message, timestamp, data
        });
    }
    
    speak(text) {
        // Text-to-speech is disabled
        return;
    }
    
    cleanTextForSpeech(text) {
        // Remove emojis and clean text for better speech synthesis
        const cleanedText = text
            // Remove all emoji characters (Unicode ranges for emojis)
            .replace(/[\u{1F600}-\u{1F64F}]/gu, '') // Emoticons
            .replace(/[\u{1F300}-\u{1F5FF}]/gu, '') // Symbols & Pictographs
            .replace(/[\u{1F680}-\u{1F6FF}]/gu, '') // Transport & Map
            .replace(/[\u{1F1E0}-\u{1F1FF}]/gu, '') // Regional indicators
            .replace(/[\u{2600}-\u{26FF}]/gu, '')   // Miscellaneous Symbols
            .replace(/[\u{2700}-\u{27BF}]/gu, '')   // Dingbats
            .replace(/[\u{1F900}-\u{1F9FF}]/gu, '') // Supplemental Symbols
            .replace(/[\u{1FA70}-\u{1FAFF}]/gu, '') // Extended Pictographs
            // Remove other symbols that sound awkward when spoken
            .replace(/[⭐✨🌟⚡🎯🎉🚀]/g, '')
            .replace(/[📊📈📉📌🔧⚙️🛠️]/g, '')
            .replace(/[❌✅⚠️ℹ️]/g, '')
            // Clean up extra spaces and formatting
            .replace(/\s+/g, ' ')
            .replace(/^\s+|\s+$/g, '')
            // Replace some text patterns that don't speak well
            .replace(/\*([^*]+)\*/g, '$1') // Remove asterisk emphasis
            .replace(/`([^`]+)`/g, '$1')   // Remove backticks
            .replace(/\n+/g, '. ')        // Replace newlines with periods
            .trim();
        
        // Debug log to see the cleaning effect
        if (text !== cleanedText) {
            console.log('Speech cleaning:', {
                original: text,
                cleaned: cleanedText
            });
        }
        
        return cleanedText;
    }
    
    showFeedbackOptions() {
        if (!this.feedbackContainer) return;
        
        this.feedbackContainer.innerHTML = `
            <div class="feedback-prompt">
                <p>How was my response?</p>
                <button class="feedback-btn good" data-feedback="good">👍 Good</button>
                <button class="feedback-btn bad" data-feedback="bad">👎 Bad</button>
                <button class="feedback-btn excellent" data-feedback="excellent">⭐ Excellent</button>
            </div>
        `;
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            this.feedbackContainer.innerHTML = '';
        }, 10000);
    }
    
    async submitFeedback(feedback) {
        try {
            await fetch('/api/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_input: this.lastInput,
                    ai_response: this.lastResponse,
                    feedback: feedback
                })
            });
            
            this.feedbackContainer.innerHTML = '<p class="feedback-thanks">Thanks for your feedback! 🙏</p>';
            setTimeout(() => {
                this.feedbackContainer.innerHTML = '';
            }, 3000);
            
        } catch (error) {
            console.error('Error submitting feedback:', error);
        }
    }
    
    async setPersonality(personality) {
        this.currentPersonality = personality;
        try {
            await fetch('/api/personality', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ personality })
            });
            
            this.addMessage('System', `Personality changed to ${personality}`, 'system');
        } catch (error) {
            console.error('Error setting personality:', error);
        }
    }
    
    async loadActiveTimersReminders() {
        try {
            const response = await fetch('/api/timers-reminders');
            if (!response.ok) {
                console.error('Failed to load timers/reminders:', response.status);
                return;
            }
            
            const data = await response.json();
            this.activeTimers = data.timers || [];
            this.activeReminders = data.reminders || [];
            
            console.log('Loaded timers:', this.activeTimers);
            console.log('Loaded reminders:', this.activeReminders);
            
            this.updateTimersDisplay();
            this.updateRemindersDisplay();
            
        } catch (error) {
            console.error('Error loading timers/reminders:', error);
        }
    }
    
    updateTimersDisplay() {
        if (!this.timersContainer) return;
        
        if (this.activeTimers.length === 0) {
            this.timersContainer.innerHTML = '<div class="no-items">No active timers</div>';
            return;
        }
        
        const timersHtml = this.activeTimers.map(timer => {
            const minutes = Math.floor(timer.remaining_seconds / 60);
            const seconds = timer.remaining_seconds % 60;
            const timeDisplay = `${minutes}:${seconds.toString().padStart(2, '0')}`;
            
            return `
                <div class="timer-item" data-timer-id="${timer.id}">
                    <div class="timer-info">
                        <span class="timer-description">${timer.description}</span>
                        <span class="timer-time">${timeDisplay}</span>
                    </div>
                    <button class="cancel-timer-btn" data-timer-id="${timer.id}">Cancel</button>
                </div>
            `;
        }).join('');
        
        this.timersContainer.innerHTML = timersHtml;
    }
    
    updateRemindersDisplay() {
        if (!this.remindersContainer) return;
        
        if (this.activeReminders.length === 0) {
            this.remindersContainer.innerHTML = '<div class="no-items">No active reminders</div>';
            return;
        }
        
        const remindersHtml = this.activeReminders.map(reminder => {
            const minutesUntil = reminder.minutes_until;
            const timeText = minutesUntil > 0 ? `in ${minutesUntil} minutes` : 'overdue';
            
            return `
                <div class="reminder-item" data-reminder-id="${reminder.id}">
                    <div class="reminder-info">
                        <span class="reminder-text">${reminder.text}</span>
                        <span class="reminder-time">${timeText}</span>
                    </div>
                    <button class="cancel-reminder-btn" data-reminder-id="${reminder.id}">Cancel</button>
                </div>
            `;
        }).join('');
        
        this.remindersContainer.innerHTML = remindersHtml;
    }
    
    async cancelTimer(timerId) {
        try {
            const response = await fetch(`/api/cancel-timer/${timerId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.ok) {
                console.log(`Timer ${timerId} cancelled`);
                // Reload timers to update display
                await this.loadActiveTimersReminders();
            } else {
                console.error('Failed to cancel timer:', response.status);
            }
        } catch (error) {
            console.error('Error cancelling timer:', error);
        }
    }
    
    async cancelReminder(reminderId) {
        try {
            const response = await fetch(`/api/cancel-reminder/${reminderId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.ok) {
                console.log(`Reminder ${reminderId} cancelled`);
                // Reload reminders to update display
                await this.loadActiveTimersReminders();
            } else {
                console.error('Failed to cancel reminder:', response.status);
            }
        } catch (error) {
            console.error('Error cancelling reminder:', error);
        }
    }
    
    updateTimersDisplay() {
        if (!this.timersContainer) return;
        
        if (this.activeTimers.length === 0) {
            this.timersContainer.innerHTML = '<p>No active timers</p>';
            return;
        }
        
        let timersHTML = '<h4>Active Timers</h4>';
        this.activeTimers.forEach(timer => {
            const remainingTime = this.formatTime(timer.remaining_seconds);
            timersHTML += `
                <div class="timer-item">
                    <span>${timer.duration} (${remainingTime} left)</span>
                    <button class="cancel-timer-btn" data-timer-id="${timer.id}">Cancel</button>
                </div>
            `;
        });
        
        this.timersContainer.innerHTML = timersHTML;
    }
    
    updateRemindersDisplay() {
        if (!this.remindersContainer) return;
        
        if (this.activeReminders.length === 0) {
            this.remindersContainer.innerHTML = '<p>No active reminders</p>';
            return;
        }
        
        let remindersHTML = '<h4>Active Reminders</h4>';
        this.activeReminders.forEach(reminder => {
            remindersHTML += `
                <div class="reminder-item">
                    <span>${reminder.text} (${reminder.time})</span>
                    <button class="cancel-reminder-btn" data-reminder-id="${reminder.id}">Cancel</button>
                </div>
            `;
        });
        
        this.remindersContainer.innerHTML = remindersHTML;
    }
    
    async cancelTimer(timerId) {
        try {
            const response = await fetch(`/api/cancel-timer/${timerId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (response.ok) {
                this.addMessage('System', 'Timer cancelled successfully', 'system');
                this.loadActiveTimersReminders();
            }
        } catch (error) {
            console.error('Error cancelling timer:', error);
        }
    }
    
    async cancelReminder(reminderId) {
        try {
            const response = await fetch(`/api/cancel-reminder/${reminderId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (response.ok) {
                this.addMessage('System', 'Reminder cancelled successfully', 'system');
                this.loadActiveTimersReminders();
            }
        } catch (error) {
            console.error('Error cancelling reminder:', error);
        }
    }
    
    formatTime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        
        if (hours > 0) {
            return `${hours}h ${minutes}m ${secs}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    }
    
    startPeriodicUpdates() {
        // Update timers and reminders every 5 seconds
        setInterval(() => {
            if (this.activeTimers.length > 0 || this.activeReminders.length > 0) {
                this.loadActiveTimersReminders();
            }
        }, 5000);
        
        // Update timer countdowns every second
        setInterval(() => {
            this.activeTimers = this.activeTimers.map(timer => {
                timer.remaining_seconds = Math.max(0, timer.remaining_seconds - 1);
                return timer;
            }).filter(timer => timer.remaining_seconds > 0);
            
            this.updateTimersDisplay();
        }, 1000);
    }
    
    updateStatus(status) {
        if (this.statusIndicator) {
            this.statusIndicator.textContent = status;
        }
        console.log('Status:', status);
    }
    
    // New: Session management functions
    async initializeSession() {
        try {
            // Check if we have an existing active session
            const response = await fetch('/api/conversation/sessions');
            const data = await response.json();
            
            if (data.sessions && data.sessions.length > 0) {
                const activeSession = data.sessions.find(s => s.is_active);
                if (activeSession) {
                    this.sessionId = activeSession.id;
                    this.conversationLength = activeSession.message_count;
                    this.currentPersonality = activeSession.personality;
                    
                    // Update personality selector
                    if (this.personalitySelect) {
                        this.personalitySelect.value = this.currentPersonality;
                    }
                    
                    // Load conversation history
                    this.loadConversationHistory();
                    return;
                }
            }
            
            // Create new session if no active session exists
            this.createNewSession();
            
        } catch (error) {
            console.error('Error initializing session:', error);
            // Continue without session if there's an error
        }
    }
    
    async createNewSession() {
        try {
            const response = await fetch('/api/conversation/new-session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    personality: this.currentPersonality
                })
            });
            
            const data = await response.json();
            this.sessionId = data.session_id;
            this.conversationLength = 0;
            this.contextUsed = false;
            
            console.log('New conversation session created:', this.sessionId);
            
        } catch (error) {
            console.error('Error creating new session:', error);
        }
    }
    
    async loadConversationHistory() {
        try {
            if (!this.sessionId) return;
            
            const response = await fetch(`/api/conversation/history?session_id=${this.sessionId}&limit=10`);
            const data = await response.json();
            
            // Clear current messages and load history
            if (this.messagesContainer) {
                this.messagesContainer.innerHTML = '';
            }
            
            // Add historical messages
            for (const msg of data.history) {
                this.addMessage('You', msg.user_input, 'user', null, false);
                this.addMessage('AI', msg.ai_response, 'ai', {
                    confidence: msg.confidence,
                    intent: msg.intent,
                    timestamp: msg.timestamp
                }, false);
            }
            
            // Update conversation count
            this.conversationLength = data.message_count;
            if (this.conversationCount) {
                this.conversationCount.textContent = this.conversationLength;
            }
            
        } catch (error) {
            console.error('Error loading conversation history:', error);
        }
    }
    
    async clearConversation() {
        try {
            // Store conversation data before clearing (if memory system is available)
            if (this.memorySystem && this.conversationHistory.length > 0) {
                await this.endConversationWithMemory();
            }
            
            await this.createNewSession();
            
            // Clear messages
            if (this.messagesContainer) {
                this.messagesContainer.innerHTML = '';
            }
            
            // Reset counters
            this.conversationLength = 0;
            if (this.conversationCount) {
                this.conversationCount.textContent = '0';
            }
            
            // Add welcome message with context awareness
            let welcomeMessage = 'Hello! I\'m Horizon, your AI assistant. How can I help you today?';
            
            // Check if we have persistent context to acknowledge
            if (this.memorySystem && this.memorySystem.persistentMemory.size > 0) {
                welcomeMessage = 'Hello again! I remember our previous conversations. How can I help you today?';
            }
            
            this.addMessage('AI', welcomeMessage, 'ai');
            
        } catch (error) {
            console.error('Error clearing conversation:', error);
        }
    }
    
    // Test microphone access
    async testMicrophone() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.addMessage('System', '✅ Microphone test successful!', 'system');
            stream.getTracks().forEach(track => track.stop()); // Stop the stream
            return true;
        } catch (error) {
            this.addMessage('System', `❌ Microphone test failed: ${error.message}`, 'error');
            return false;
        }
    }
    
    // Simple Horizon explanation help - now appears in chat
    showHorizonHelp() {
        // Add help information as chat messages
        this.addMessage('You', 'Help - What is Horizon?', 'user');
        
        this.addMessage('Horizon AI', `Horizon is a next-generation AI chatbot that goes beyond conversation—combining ChatGPT-style intelligence with fast-action features like quick commands, instant timers, and smart reminders. Designed for both productivity and natural interaction, Horizon delivers lightning-fast responses while helping users stay organized and in control.`, 'ai');
    }
    
    // ===== MEMORY & LEARNING INTEGRATION =====
    
    // Integration with memory system
    setupMemoryIntegration() {
        if (window.memoryLearning) {
            console.log('🔗 Integrating with Memory & Learning system');
            this.memorySystem = window.memoryLearning;
            
            // Set up event listeners for memory integration
            this.addEventListener('conversation_ended', (event) => {
                if (this.memorySystem) {
                    this.memorySystem.onConversationEnd(event.detail);
                }
            });
        }
    }
    
    // Integration with contextual intelligence
    setupContextualIntegration() {
        if (window.contextualIntelligence) {
            console.log('🌍 Integrating with Contextual Intelligence system');
            this.contextualIntelligence = window.contextualIntelligence;
        }
    }
    
    // Get contextual data for API requests
    getContextualData() {
        if (!this.contextualIntelligence) return null;
        
        return {
            location: this.contextualIntelligence.getLocationContext(),
            time: this.contextualIntelligence.getTimeContext(),
            weather: this.contextualIntelligence.getWeatherContext(),
            suggestions: this.contextualIntelligence.getContextualSuggestions()
        };
    }
    
    // Update personal context for memory system
    updatePersonalContext(context) {
        console.log('📝 Updating personal context:', context);
        
        // Store in memory system if available
        if (this.memorySystem) {
            this.memorySystem.storePersistentContext({
                context_type: 'personal_info',
                context_key: context.key || 'personal_data',
                context_value: context,
                importance_score: 0.9,
                decay_rate: 0.01 // Very slow decay for personal info
            });
        }
    }
    
    // Record important conversation facts
    recordConversationFact(fact, importance = 0.7) {
        if (this.memorySystem) {
            this.memorySystem.storePersistentContext({
                context_type: 'conversation_fact',
                context_key: `fact_${Date.now()}`,
                context_value: { fact: fact, timestamp: new Date().toISOString() },
                importance_score: importance,
                decay_rate: 0.05
            });
        }
    }
    
    // Learn from user feedback
    learnFromFeedback(feedback, responseContext) {
        if (this.memorySystem) {
            this.memorySystem.recordPreferenceFeedback({
                feedback_type: feedback.type, // 'positive', 'negative', 'correction'
                interaction_context: responseContext,
                user_feedback: feedback.message,
                response_quality_rating: feedback.rating || 0.5,
                feedback_category: feedback.category || 'general'
            });
        }
    }
    
    // Enhanced conversation ending with memory storage
    async endConversationWithMemory() {
        const conversationData = {
            important_facts: this.extractImportantFacts(),
            unresolved_questions: this.findUnresolvedQuestions(),
            user_preferences: this.inferredPreferences(),
            summary: this.generateConversationSummary(),
            session_id: this.sessionId,
            conversation_length: this.conversationLength
        };
        
        // Dispatch event for memory system
        const event = new CustomEvent('conversation_ended', {
            detail: conversationData
        });
        this.dispatchEvent(event);
        
        // Store conversation summary
        if (this.memorySystem) {
            this.memorySystem.storeConversationMemory({
                memory_type: 'session_summary',
                memory_content: conversationData.summary,
                memory_summary: `Session with ${this.conversationLength} exchanges`,
                relevance_score: this.conversationLength > 5 ? 0.8 : 0.4
            });
        }
    }
    
    // Extract important facts from conversation
    extractImportantFacts() {
        const facts = [];
        const keywords = ['my name is', 'i am', 'i work at', 'i live in', 'i like', 'i prefer', 'remember that'];
        
        this.conversationHistory.forEach(message => {
            if (message.type === 'user') {
                const lowerText = message.text.toLowerCase();
                keywords.forEach(keyword => {
                    if (lowerText.includes(keyword)) {
                        facts.push({
                            fact: message.text,
                            timestamp: message.timestamp || new Date().toISOString(),
                            confidence: 0.8
                        });
                    }
                });
            }
        });
        
        return facts;
    }
    
    // Find unresolved questions
    findUnresolvedQuestions() {
        const questions = [];
        
        this.conversationHistory.forEach(message => {
            if (message.text.includes('?') && message.type === 'user') {
                // Check if there's a follow-up that indicates resolution
                const messageIndex = this.conversationHistory.indexOf(message);
                const nextMessages = this.conversationHistory.slice(messageIndex + 1, messageIndex + 3);
                
                const hasResolution = nextMessages.some(msg => 
                    msg.type === 'user' && 
                    (msg.text.toLowerCase().includes('thank') || 
                     msg.text.toLowerCase().includes('got it') ||
                     msg.text.toLowerCase().includes('perfect'))
                );
                
                if (!hasResolution) {
                    questions.push({
                        question: message.text,
                        timestamp: message.timestamp || new Date().toISOString(),
                        context: nextMessages.map(m => m.text).join(' ')
                    });
                }
            }
        });
        
        return questions;
    }
    
    // Infer user preferences from conversation
    inferredPreferences() {
        const preferences = {};
        
        // Analyze response lengths preference
        const userMessages = this.conversationHistory.filter(m => m.type === 'user');
        const avgUserLength = userMessages.reduce((sum, m) => sum + m.text.length, 0) / userMessages.length;
        
        if (avgUserLength < 50) {
            preferences.communication_style = 'concise';
        } else if (avgUserLength > 200) {
            preferences.communication_style = 'detailed';
        } else {
            preferences.communication_style = 'balanced';
        }
        
        // Analyze topic interests
        const topics = ['technology', 'science', 'business', 'creative', 'personal', 'learning'];
        topics.forEach(topic => {
            const mentions = this.conversationHistory.filter(m => 
                m.text.toLowerCase().includes(topic)
            ).length;
            
            if (mentions > 0) {
                preferences[`interest_${topic}`] = Math.min(mentions / 10, 1.0);
            }
        });
        
        return preferences;
    }
    
    // Generate conversation summary
    generateConversationSummary() {
        const totalMessages = this.conversationHistory.length;
        const userMessages = this.conversationHistory.filter(m => m.type === 'user').length;
        const aiMessages = this.conversationHistory.filter(m => m.type === 'ai').length;
        
        const topics = this.extractMainTopics();
        const duration = this.calculateSessionDuration();
        
        return {
            total_exchanges: Math.floor(totalMessages / 2),
            user_messages: userMessages,
            ai_messages: aiMessages,
            main_topics: topics,
            session_duration_minutes: duration,
            engagement_level: this.calculateEngagementLevel()
        };
    }
    
    extractMainTopics() {
        // Simple topic extraction based on keywords
        const topicKeywords = {
            technical: ['code', 'programming', 'software', 'technology', 'computer'],
            creative: ['design', 'art', 'music', 'creative', 'writing'],
            personal: ['personal', 'life', 'family', 'friends', 'relationship'],
            business: ['work', 'job', 'business', 'career', 'professional'],
            learning: ['learn', 'study', 'education', 'course', 'tutorial']
        };
        
        const allText = this.conversationHistory.map(m => m.text.toLowerCase()).join(' ');
        const topics = [];
        
        Object.entries(topicKeywords).forEach(([topic, keywords]) => {
            const mentions = keywords.filter(keyword => allText.includes(keyword)).length;
            if (mentions > 0) {
                topics.push({ topic, relevance: Math.min(mentions / 5, 1.0) });
            }
        });
        
        return topics.sort((a, b) => b.relevance - a.relevance).slice(0, 3);
    }
    
    calculateSessionDuration() {
        if (this.conversationHistory.length < 2) return 0;
        
        const firstMessage = this.conversationHistory[0];
        const lastMessage = this.conversationHistory[this.conversationHistory.length - 1];
        
        if (firstMessage.timestamp && lastMessage.timestamp) {
            const start = new Date(firstMessage.timestamp);
            const end = new Date(lastMessage.timestamp);
            return Math.round((end - start) / (1000 * 60)); // minutes
        }
        
        return 0;
    }
    
    calculateEngagementLevel() {
        // Calculate engagement based on message frequency and length
        const avgResponseTime = this.calculateAverageResponseTime();
        const avgMessageLength = this.conversationHistory.reduce((sum, m) => sum + m.text.length, 0) / this.conversationHistory.length;
        
        let engagement = 0.5; // baseline
        
        // Quick responses indicate high engagement
        if (avgResponseTime < 30000) engagement += 0.2; // < 30 seconds
        if (avgResponseTime < 10000) engagement += 0.1; // < 10 seconds
        
        // Longer messages indicate engagement
        if (avgMessageLength > 100) engagement += 0.1;
        if (avgMessageLength > 200) engagement += 0.1;
        
        // Conversation length indicates engagement
        if (this.conversationHistory.length > 10) engagement += 0.1;
        if (this.conversationHistory.length > 20) engagement += 0.1;
        
        return Math.min(engagement, 1.0);
    }
    
    calculateAverageResponseTime() {
        let totalTime = 0;
        let pairs = 0;
        
        for (let i = 1; i < this.conversationHistory.length; i++) {
            const current = this.conversationHistory[i];
            const previous = this.conversationHistory[i - 1];
            
            if (current.timestamp && previous.timestamp) {
                totalTime += new Date(current.timestamp) - new Date(previous.timestamp);
                pairs++;
            }
        }
        
        return pairs > 0 ? totalTime / pairs : 0;
    }
    
    // EventTarget implementation for dispatching events
    addEventListener(type, listener) {
        if (!this._eventListeners) this._eventListeners = {};
        if (!this._eventListeners[type]) this._eventListeners[type] = [];
        this._eventListeners[type].push(listener);
    }
    
    removeEventListener(type, listener) {
        if (!this._eventListeners || !this._eventListeners[type]) return;
        const index = this._eventListeners[type].indexOf(listener);
        if (index > -1) {
            this._eventListeners[type].splice(index, 1);
        }
    }
    
    dispatchEvent(event) {
        if (!this._eventListeners || !this._eventListeners[event.type]) return;
        this._eventListeners[event.type].forEach(listener => {
            try {
                listener(event);
            } catch (error) {
                console.error('Error in event listener:', error);
            }
        });
    }
}

// Initialize the enhanced AI assistant when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.aiAssistant = new EnhancedAIVoiceAssistant();
    
    // Add help button functionality
    const helpBtn = document.getElementById('helpBtn');
    if (helpBtn) {
        helpBtn.addEventListener('click', () => {
            window.aiAssistant.showHorizonHelp();
        });
    }
    
    // Set up memory integration when memory system is available
    setTimeout(() => {
        if (window.memoryLearning) {
            window.aiAssistant.setupMemoryIntegration();
        }
    }, 2000); // Wait for memory system to load
    
    // Set up contextual intelligence integration 
    setTimeout(() => {
        if (window.contextualIntelligence) {
            window.aiAssistant.setupContextualIntegration();
        }
    }, 3000); // Wait for contextual system to load
    
    // Initialize educational features
    initializeEducationalFeatures();
});

// ===== EDUCATIONAL FEATURES JAVASCRIPT =====

function initializeEducationalFeatures() {
    // Initialize form submissions
    initializeCurriculumForms();
    initializeLanguageForms();
    
    // Load existing data
    loadEducationalDashboardData();
}

function initializeCurriculumForms() {
    // Create Curriculum Form
    const createCurriculumForm = document.getElementById('createCurriculumForm');
    if (createCurriculumForm) {
        createCurriculumForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(createCurriculumForm);
            const curriculumData = {
                subject: formData.get('subject'),
                grade_level: formData.get('grade_level'),
                description: formData.get('description')
            };
            
            try {
                showEducationalLoading('Creating your personalized curriculum...');
                
                const response = await fetch('/api/curriculums', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(curriculumData)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    showToast('success', 'Curriculum Created!', 'Your personalized curriculum has been successfully created.');
                    closeModal('createCurriculumModal');
                    createCurriculumForm.reset();
                    loadMyCurriculums();
                } else {
                    showToast('error', 'Creation Failed', result.error || 'Failed to create curriculum');
                }
            } catch (error) {
                console.error('Error creating curriculum:', error);
                showToast('error', 'Network Error', 'Please check your connection and try again');
            } finally {
                hideEducationalLoading();
            }
        });
    }
}

function initializeLanguageForms() {
    // Practice Conversation Form
    const practiceForm = document.getElementById('practiceConversationForm');
    if (practiceForm) {
        practiceForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(practiceForm);
            const sessionData = {
                language: formData.get('language'),
                level: formData.get('level'),
                topic: formData.get('topic')
            };
            
            try {
                showEducationalLoading('Starting your language conversation session...');
                
                const response = await fetch('/api/language-sessions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(sessionData)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    showToast('success', 'Session Started!', `Your ${sessionData.language} conversation practice has begun.`);
                    closeModal('practiceConversationModal');
                    practiceForm.reset();
                    startConversationInterface(result);
                } else {
                    showToast('error', 'Session Failed', result.error || 'Failed to start language session');
                }
            } catch (error) {
                console.error('Error starting language session:', error);
                showToast('error', 'Network Error', 'Please check your connection and try again');
            } finally {
                hideEducationalLoading();
            }
        });
    }
}

async function loadMyCurriculums() {
    try {
        const response = await fetch('/api/curriculums');
        const data = await response.json();
        
        const container = document.getElementById('myCurriculumsContent');
        if (!container) return;
        
        if (data.curriculums && data.curriculums.length > 0) {
            container.innerHTML = data.curriculums.map(curriculum => createCurriculumCard(curriculum)).join('');
        } else {
            container.innerHTML = `
                <div class="educational-loading">
                    <p class="educational-loading-text">No curriculums found</p>
                    <button class="btn-card primary" onclick="openModal('createCurriculumModal')">
                        Create Your First Curriculum
                    </button>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading curriculums:', error);
        showToast('error', 'Loading Failed', 'Unable to load your curriculums');
    }
}

async function loadBrowseCurriculums() {
    try {
        const response = await fetch('/api/curriculums?browse=true');
        const data = await response.json();
        
        const container = document.getElementById('browseCurriculumsContent');
        if (!container) return;
        
        if (data.curriculums && data.curriculums.length > 0) {
            container.innerHTML = data.curriculums.map(curriculum => createBrowseCurriculumCard(curriculum)).join('');
        } else {
            container.innerHTML = `
                <div class="educational-loading">
                    <p class="educational-loading-text">No public curriculums available</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading browse curriculums:', error);
        showToast('error', 'Loading Failed', 'Unable to load curriculum library');
    }
}

function createCurriculumCard(curriculum) {
    const progress = curriculum.progress || 0;
    return `
        <div class="curriculum-card">
            <h3>${curriculum.subject} - ${curriculum.grade_level}</h3>
            <div class="curriculum-meta">
                <span class="curriculum-subject">${curriculum.subject}</span>
                <span class="curriculum-grade">${curriculum.grade_level}</span>
            </div>
            <p class="curriculum-description">${curriculum.description}</p>
            <div class="curriculum-progress">
                <div class="curriculum-progress-bar" style="width: ${progress}%"></div>
            </div>
            <div class="curriculum-actions">
                <button class="curriculum-btn" onclick="continueCurriculum(${curriculum.id})">
                    📚 Continue Learning
                </button>
                <button class="curriculum-btn" onclick="viewCurriculumDetails(${curriculum.id})">
                    📊 View Progress
                </button>
            </div>
        </div>
    `;
}

function createBrowseCurriculumCard(curriculum) {
    return `
        <div class="curriculum-card">
            <h3>${curriculum.subject} - ${curriculum.grade_level}</h3>
            <div class="curriculum-meta">
                <span class="curriculum-subject">${curriculum.subject}</span>
                <span class="curriculum-grade">${curriculum.grade_level}</span>
            </div>
            <p class="curriculum-description">${curriculum.description}</p>
            <div class="curriculum-actions">
                <button class="curriculum-btn" onclick="adoptCurriculum(${curriculum.id})">
                    ➕ Add to My Learning
                </button>
                <button class="curriculum-btn" onclick="previewCurriculum(${curriculum.id})">
                    👁️ Preview
                </button>
            </div>
        </div>
    `;
}

async function loadVocabularyBuilder() {
    try {
        const response = await fetch('/api/vocabulary');
        const data = await response.json();
        
        const container = document.getElementById('vocabularyBuilderContent');
        if (!container) return;
        
        if (data.words && data.words.length > 0) {
            container.innerHTML = data.words.map(word => createVocabularyExercise(word)).join('');
        } else {
            container.innerHTML = `
                <div class="educational-loading">
                    <p class="educational-loading-text">Start practicing vocabulary</p>
                    <button class="btn-card primary" onclick="generateVocabularyExercises()">
                        Generate New Words
                    </button>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading vocabulary:', error);
        showToast('error', 'Loading Failed', 'Unable to load vocabulary exercises');
    }
}

function createVocabularyExercise(word) {
    return `
        <div class="vocabulary-exercise">
            <div class="vocabulary-word">${word.word}</div>
            <div class="vocabulary-pronunciation">[${word.pronunciation || 'N/A'}]</div>
            <div class="vocabulary-definition">${word.definition}</div>
            <div class="vocabulary-example">"${word.example}"</div>
            <div class="vocabulary-actions">
                <button class="vocab-btn" onclick="markWordMastered(${word.id})">
                    ✅ Mastered
                </button>
                <button class="vocab-btn" onclick="hearPronunciation('${word.word}')">
                    🔊 Listen
                </button>
                <button class="vocab-btn" onclick="practiceWord(${word.id})">
                    📝 Practice
                </button>
            </div>
        </div>
    `;
}

async function loadLanguageProgress() {
    try {
        const response = await fetch('/api/language-progress');
        const data = await response.json();
        
        const container = document.getElementById('languageProgressContent');
        if (!container) return;
        
        if (data.progress) {
            container.innerHTML = createLanguageProgressDisplay(data.progress);
        } else {
            container.innerHTML = `
                <div class="educational-loading">
                    <p class="educational-loading-text">No language progress data available</p>
                    <button class="btn-card primary" onclick="openModal('practiceConversationModal')">
                        Start Language Practice
                    </button>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading language progress:', error);
        showToast('error', 'Loading Failed', 'Unable to load language progress');
    }
}

function createLanguageProgressDisplay(progress) {
    return `
        <div class="progress-metrics">
            <div class="progress-metric">
                <span class="metric-value">${progress.sessions_completed || 0}</span>
                <span class="metric-label">Sessions Completed</span>
            </div>
            <div class="progress-metric">
                <span class="metric-value">${progress.vocabulary_mastered || 0}</span>
                <span class="metric-label">Words Mastered</span>
            </div>
            <div class="progress-metric">
                <span class="metric-value">${progress.fluency_score || 0}%</span>
                <span class="metric-label">Fluency Score</span>
            </div>
            <div class="progress-metric">
                <span class="metric-value">${progress.streak_days || 0}</span>
                <span class="metric-label">Day Streak</span>
            </div>
        </div>
        <div class="fluency-indicator">
            <span>Current Level:</span>
            <span class="fluency-level ${progress.current_level || 'A1'}">${progress.current_level || 'A1'}</span>
        </div>
    `;
}

async function loadEducationalDashboardData() {
    try {
        // Update curriculum count
        const curriculumsResponse = await fetch('/api/curriculums');
        const curriculumsData = await curriculumsResponse.json();
        
        const curriculumCount = document.getElementById('curriculumCount');
        if (curriculumCount) {
            curriculumCount.textContent = curriculumsData.curriculums?.length || 0;
        }
        
        const curriculumProgress = document.getElementById('curriculumProgress');
        if (curriculumProgress && curriculumsData.curriculums?.length > 0) {
            const avgProgress = curriculumsData.curriculums.reduce((sum, curr) => sum + (curr.progress || 0), 0) / curriculumsData.curriculums.length;
            curriculumProgress.textContent = Math.round(avgProgress) + '%';
        }
        
        // Update language learning stats
        const languageResponse = await fetch('/api/language-progress');
        const languageData = await languageResponse.json();
        
        const languageCount = document.getElementById('languageCount');
        if (languageCount) {
            languageCount.textContent = languageData.languages?.length || 0;
        }
        
        const fluencyLevel = document.getElementById('fluencyLevel');
        if (fluencyLevel) {
            fluencyLevel.textContent = languageData.progress?.current_level || 'A1';
        }
        
    } catch (error) {
        console.error('Error loading educational dashboard data:', error);
    }
}

// Educational Feature Action Functions
function continueCurriculum(curriculumId) {
    showToast('info', 'Continuing Curriculum', 'Loading your curriculum progress...');
    // Implementation for continuing curriculum
}

function viewCurriculumDetails(curriculumId) {
    showToast('info', 'Loading Details', 'Fetching curriculum analytics...');
    // Implementation for viewing curriculum details
}

function adoptCurriculum(curriculumId) {
    showToast('info', 'Adding Curriculum', 'Adding to your learning path...');
    // Implementation for adopting a curriculum
}

function previewCurriculum(curriculumId) {
    showToast('info', 'Preview Loading', 'Preparing curriculum preview...');
    // Implementation for previewing curriculum
}

function markWordMastered(wordId) {
    showToast('success', 'Word Mastered!', 'Great job! This word has been marked as mastered.');
    // Implementation for marking word as mastered
}

function hearPronunciation(word) {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(word);
        speechSynthesis.speak(utterance);
        showToast('info', 'Pronunciation', `Playing pronunciation for "${word}"`);
    } else {
        showToast('error', 'Not Supported', 'Speech synthesis not supported in this browser');
    }
}

function practiceWord(wordId) {
    showToast('info', 'Practice Mode', 'Loading word practice exercises...');
    // Implementation for word practice
}

function generateVocabularyExercises() {
    showToast('info', 'Generating', 'Creating new vocabulary exercises...');
    // Implementation for generating vocabulary
}

function startConversationInterface(sessionData) {
    // Implementation for starting conversation interface
    showToast('success', 'Conversation Started', 'Your language practice session is now active!');
}

// Modal management for educational features
function openEducationalModal(modalId) {
    openModal(modalId);
    
    // Load content based on modal type
    switch(modalId) {
        case 'myCurriculumsModal':
            loadMyCurriculums();
            break;
        case 'browseCurriculumsModal':
            loadBrowseCurriculums();
            break;
        case 'vocabularyBuilderModal':
            loadVocabularyBuilder();
            break;
        case 'languageProgressModal':
            loadLanguageProgress();
            break;
    }
}

// Utility functions for educational features
function showEducationalLoading(message) {
    showToast('info', 'Loading', message);
}

function hideEducationalLoading() {
    // Hide any loading indicators
}

// Toast notification system (if not already implemented)
function showToast(type, title, message) {
    // Create toast notification
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <div class="toast-header">
            <div class="toast-title">
                <span class="toast-icon">${getToastIcon(type)}</span>
                ${title}
            </div>
            <button class="toast-close" onclick="closeToast(this)">&times;</button>
        </div>
        <p class="toast-message">${message}</p>
    `;
    
    // Add to container
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    
    container.appendChild(toast);
    
    // Show toast
    setTimeout(() => toast.classList.add('show'), 100);
    
    // Auto-hide after 5 seconds
    setTimeout(() => closeToast(toast.querySelector('.toast-close')), 5000);
}

function getToastIcon(type) {
    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };
    return icons[type] || 'ℹ️';
}

function closeToast(button) {
    const toast = button.closest('.toast');
    toast.classList.add('hide');
    setTimeout(() => toast.remove(), 300);
}
