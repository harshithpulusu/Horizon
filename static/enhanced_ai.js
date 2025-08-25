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
        
        // Feedback buttons (will be added dynamically)
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
    }
    
    async sendMessage() {
        const input = this.voiceInput.value.trim();
        if (!input) return;
        
        this.addMessage('You', input, 'user');
        this.voiceInput.value = '';
        await this.processInput(input);
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
        
        // Update statistics
        this.confidenceScore.textContent = Math.round((data.confidence || 0.8) * 100) + '%';
        this.conversationCount.textContent = data.conversation_count || 0;
        
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
        
        // Add extra info for AI responses
        if (data && type === 'ai') {
            messageContent += `
                <div class="message-metadata">
                    <small>
                        Intent: ${data.intent} | 
                        Confidence: ${Math.round((data.confidence || 0) * 100)}% |
                        Sentiment: ${data.sentiment_analysis?.label || 'neutral'}
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
            
            const utterance = new SpeechSynthesisUtterance(text);
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
