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
        this.isWakeWordMode = false;  // New: wake word listening mode
        this.wakeWords = ['hey horizon', 'horizon', 'hey assistant', 'assistant'];  // New: wake words
        this.wakeWordSensitivity = 0.7;  // New: sensitivity threshold
        
        this.init();
        this.initEventListeners();
        this.loadActiveTimersReminders();
        this.startPeriodicUpdates();
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
        this.initWakeWordDetection();  // New: Initialize wake word detection
        
        this.updateStatus('Ready');
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
                if (event.error !== 'aborted' && this.isWakeWordMode) {
                    setTimeout(() => this.startWakeWordDetection(), 1000);
                }
            };
            
            this.wakeWordRecognition.onend = () => {
                // Restart wake word detection if it should be running
                if (this.isWakeWordMode && !this.isListening) {
                    setTimeout(() => this.startWakeWordDetection(), 100);
                }
            };
            
            console.log('Wake word detection initialized');
            this.addWakeWordToggle();  // Add UI toggle for wake word mode
            
        } catch (error) {
            console.error('Error initializing wake word detection:', error);
        }
    }
    
    addWakeWordToggle() {
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
                this.cancelTimer(e.target.dataset.timerId);
            } else if (e.target.classList.contains('cancel-reminder-btn')) {
                this.cancelReminder(e.target.dataset.reminderId);
            }
        });
    }
    
    initQuickCommandListeners() {
        // Add listeners to quick command buttons
        const quickCmdButtons = document.querySelectorAll('.quick-cmd-btn');
        console.log('Found quick command buttons:', quickCmdButtons.length);
        
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
        
        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    input: input,
                    personality: this.currentPersonality
                })
            });
            
            if (!response.ok) throw new Error('Network response was not ok');
            
            const data = await response.json();
            this.handleAIResponse(data);
            
        } catch (error) {
            console.error('Error processing input:', error);
            this.addMessage('AI', 'Sorry, I encountered an error processing your request.', 'error');
            this.updateStatus('Error');
        }
    }
    
    handleAIResponse(data) {
        this.lastResponse = data.response;
        this.addMessage('AI', data.response, 'ai', data);
        this.speak(data.response);
        
        // Update statistics with realistic values
        const confidence = Math.round((data.confidence || 0.82) * 100);
        this.confidenceScore.textContent = confidence + '%';
        this.conversationCount.textContent = (parseInt(this.conversationCount.textContent) + 1);
        
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
    
    addMessage(sender, message, type, data = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const timestamp = new Date().toLocaleTimeString();
        let messageContent = `
            <div class="message-header">
                <strong>${sender}</strong>
                <span class="timestamp">${timestamp}</span>
            </div>
            <div class="message-content">${message}</div>
        `;
        
        // Add enhanced metadata for AI responses
        if (data && type === 'ai') {
            const confidence = data.confidence || 0.82;
            const aiSource = data.ai_source || 'fallback';
            const responseTime = data.response_time || '0.8s';
            const intent = data.intent || 'general';
            
            messageContent += `
                <div class="message-metadata">
                    <small>
                        Response Time: ${responseTime} | 
                        Confidence: ${Math.round(confidence * 100)}% |
                        Source: ${aiSource === 'chatgpt' ? 'ChatGPT API' : 'Smart Fallback'} |
                        Intent: ${intent}
                    </small>
                </div>
            `;
        }
        
        messageDiv.innerHTML = messageContent;
        this.messagesContainer.appendChild(messageDiv);
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        
        // Store in conversation history
        this.conversationHistory.push({
            sender, message, timestamp, data
        });
    }
    
    speak(text) {
        if (this.synthesis && !this.isSpeaking) {
            // Cancel any ongoing speech
            this.synthesis.cancel();
            
            // Clean text by removing emojis and extra symbols for speech
            const cleanText = this.cleanTextForSpeech(text);
            
            const utterance = new SpeechSynthesisUtterance(cleanText);
            utterance.rate = 0.9;
            utterance.pitch = 1.0;
            utterance.volume = 0.8;
            
            utterance.onstart = () => {
                this.isSpeaking = true;
                this.updateStatus('Speaking...');
            };
            
            utterance.onend = () => {
                this.isSpeaking = false;
                this.updateStatus('Ready');
            };
            
            utterance.onerror = () => {
                this.isSpeaking = false;
                this.updateStatus('Ready');
            };
            
            this.synthesis.speak(utterance);
        }
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
            if (!response.ok) return;
            
            const data = await response.json();
            this.activeTimers = data.timers || [];
            this.activeReminders = data.reminders || [];
            
            this.updateTimersDisplay();
            this.updateRemindersDisplay();
            
        } catch (error) {
            console.error('Error loading timers/reminders:', error);
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
            const response = await fetch('/api/cancel-timer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ timer_id: timerId })
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
            const response = await fetch('/api/cancel-reminder', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ reminder_id: reminderId })
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
    
    // Voice commands help
    showVoiceCommands() {
        fetch('/api/voice-commands')
            .then(response => response.json())
            .then(commands => {
                let commandsHTML = '<div class="voice-commands-help"><h3>Voice Commands</h3>';
                
                Object.entries(commands).forEach(([category, commandList]) => {
                    commandsHTML += `<h4>${category.replace('_', ' ').toUpperCase()}</h4><ul>`;
                    commandList.forEach(command => {
                        commandsHTML += `<li>"${command}"</li>`;
                    });
                    commandsHTML += '</ul>';
                });
                
                commandsHTML += '<p><strong>Troubleshooting:</strong></p>';
                commandsHTML += '<ul>';
                commandsHTML += '<li>✅ Make sure you allow microphone access</li>';
                commandsHTML += '<li>✅ Use Chrome, Safari, or Edge (Firefox may not work)</li>';
                commandsHTML += '<li>✅ Check that you\'re on localhost or HTTPS</li>';
                commandsHTML += '<li>✅ Speak clearly and wait for the listening indicator</li>';
                commandsHTML += '</ul>';
                commandsHTML += '</div>';
                
                // Show in a modal or dedicated area
                const helpContainer = document.getElementById('helpContainer');
                if (helpContainer) {
                    helpContainer.innerHTML = commandsHTML;
                }
            })
            .catch(error => {
                console.error('Error loading voice commands:', error);
                const helpContainer = document.getElementById('helpContainer');
                if (helpContainer) {
                    helpContainer.innerHTML = '<p>Error loading voice commands. Please check your connection.</p>';
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
            window.aiAssistant.showVoiceCommands();
        });
    }
});
