// Enhanced AI Voice Assistant with advanced features

class EnhancedAIVoiceAssistant {
    constructor() {
        this.isListening = false;
        this.isSpeaking = false;
        this.recognition = null;
        this.synthesis = window.speechSynthesis;
        this.currentPersonality = 'friendly';
        this.conversationHistory = [];
        this.activeTimers = [];
        this.activeReminders = [];
        this.lastResponse = '';
        this.lastInput = '';
        
        this.init();
        this.initEventListeners();
        this.loadActiveTimersReminders();
        this.startPeriodicUpdates();
    }
    
    init() {
        // Initialize speech recognition
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
            this.recognition = new SpeechRecognition();
            this.recognition.continuous = false;
            this.recognition.interimResults = false;
            this.recognition.lang = 'en-US';
            
            this.recognition.onresult = (event) => this.handleSpeechResult(event);
            this.recognition.onerror = (event) => this.handleSpeechError(event);
            this.recognition.onend = () => this.handleSpeechEnd();
        }
        
        // Get DOM elements
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
        
        this.updateStatus('Ready');
    }
    
    initEventListeners() {
        // Voice controls
        this.startBtn?.addEventListener('click', () => this.startListening());
        this.stopBtn?.addEventListener('click', () => this.stopListening());
        this.sendBtn?.addEventListener('click', () => this.sendMessage());
        
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
            this.addMessage('AI', 'Speech recognition not supported in this browser.', 'error');
            return;
        }
        
        try {
            this.isListening = true;
            this.updateStatus('Listening...');
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.recognition.start();
            
            // Add visual feedback
            this.statusIndicator?.classList.add('listening');
        } catch (error) {
            console.error('Speech recognition error:', error);
            this.updateStatus('Error starting speech recognition');
            this.resetListeningState();
        }
    }
    
    stopListening() {
        if (this.recognition && this.isListening) {
            this.recognition.stop();
        }
        this.resetListeningState();
    }
    
    resetListeningState() {
        this.isListening = false;
        this.updateStatus('Ready');
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.statusIndicator?.classList.remove('listening');
    }
    
    handleSpeechResult(event) {
        const transcript = event.results[0][0].transcript;
        this.voiceInput.value = transcript;
        this.addMessage('You', transcript, 'user');
        this.processInput(transcript);
    }
    
    handleSpeechError(event) {
        console.error('Speech recognition error:', event.error);
        this.updateStatus(`Speech error: ${event.error}`);
        this.resetListeningState();
    }
    
    handleSpeechEnd() {
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
                
                commandsHTML += '</div>';
                
                // Show in a modal or dedicated area
                const helpContainer = document.getElementById('helpContainer');
                if (helpContainer) {
                    helpContainer.innerHTML = commandsHTML;
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
