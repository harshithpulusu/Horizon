/**
 * Chat Module - Handles messaging, conversation flow, and chat UI
 * Part of Horizon AI Assistant modular architecture
 */

class ChatModule {
    constructor() {
        // Message history for navigation (up/down arrows)
        this.messageHistory = JSON.parse(localStorage.getItem('horizonMessageHistory') || '[]');
        this.historyIndex = -1;
        
        // Conversation counter
        this.conversationCounter = 0;
        
        // DOM element references (will be set by main app)
        this.userInput = null;
        this.chatMessages = null;
        this.statusIndicator = null;
        this.conversationCount = null;
        this.personalitySelect = null;
    }

    /**
     * Initialize the chat module with DOM references
     */
    init(elements) {
        this.userInput = elements.userInput;
        this.chatMessages = elements.chatMessages;
        this.statusIndicator = elements.statusIndicator;
        this.conversationCount = elements.conversationCount;
        this.personalitySelect = elements.personalitySelect;
        
        this.setupEventListeners();
    }

    /**
     * Setup event listeners for chat functionality
     */
    setupEventListeners() {
        // Auto-resize textarea
        this.userInput.addEventListener('input', () => {
            this.userInput.style.height = 'auto';
            this.userInput.style.height = this.userInput.scrollHeight + 'px';
        });

        // Send message on Enter (but allow Shift+Enter for new lines)
        this.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
            // Message history navigation
            else if (e.key === 'ArrowUp' && this.userInput.value === '') {
                e.preventDefault();
                this.navigateHistory('up');
            } else if (e.key === 'ArrowDown' && this.userInput.value === '') {
                e.preventDefault();
                this.navigateHistory('down');
            }
        });
    }

    /**
     * Navigate through message history with up/down arrows
     */
    navigateHistory(direction) {
        if (this.messageHistory.length === 0) return;

        if (direction === 'up') {
            if (this.historyIndex < this.messageHistory.length - 1) {
                this.historyIndex++;
                this.userInput.value = this.messageHistory[this.messageHistory.length - 1 - this.historyIndex];
            }
        } else if (direction === 'down') {
            if (this.historyIndex > 0) {
                this.historyIndex--;
                this.userInput.value = this.messageHistory[this.messageHistory.length - 1 - this.historyIndex];
            } else if (this.historyIndex === 0) {
                this.historyIndex = -1;
                this.userInput.value = '';
            }
        }
    }

    /**
     * Add message to history and localStorage
     */
    addToHistory(message) {
        this.messageHistory.push(message);
        // Keep only last 15 messages
        if (this.messageHistory.length > 15) {
            this.messageHistory.shift();
        }
        localStorage.setItem('horizonMessageHistory', JSON.stringify(this.messageHistory));
        this.historyIndex = -1; // Reset history navigation
    }

    /**
     * Send message to server
     */
    async sendMessage() {
        const message = this.userInput.value.trim();
        if (!message) return;

        // Add to message history
        this.addToHistory(message);

        // Check for wake word (if voice module is available)
        if (window.voiceModule) {
            window.voiceModule.checkWakeWord(message);
        }

        // Check for local timer/reminder processing (if timer module is available)
        if (window.timerModule && window.timerModule.processTimerReminders(message)) {
            // Add user message
            this.addMessage(message, 'user');
            
            // Clear input
            this.clearInput();
            
            // Add confirmation message
            if (message.toLowerCase().includes('timer')) {
                this.addMessage('✅ Timer set and started! You can see it in the sidebar.', 'ai');
            } else if (message.toLowerCase().includes('remind')) {
                this.addMessage('✅ Reminder set! I\\'ll notify you in 5 minutes.', 'ai');
            }
            
            this.updateConversationCounter();
            return;
        }

        // Add user message
        this.addMessage(message, 'user');
        
        // Clear input
        this.clearInput();
        
        // Update status
        this.statusIndicator.textContent = 'Processing...';
        
        // Show typing indicator
        this.showTypingIndicator();

        try {
            // Send to server
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    message: message,
                    personality: this.personalitySelect.value
                })
            });

            const data = await response.json();
            
            this.hideTypingIndicator();
            this.addMessage(data.response, 'ai');
            
            // Update stats
            this.updateConversationCounter();
            this.statusIndicator.textContent = 'Ready';

        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage('Sorry, I encountered an error. Please try again.', 'ai');
            this.statusIndicator.textContent = 'Error';
            console.error('Error:', error);
        }
    }

    /**
     * Clear input and reset height
     */
    clearInput() {
        this.userInput.value = '';
        this.userInput.style.height = 'auto';
    }

    /**
     * Update conversation counter
     */
    updateConversationCounter() {
        this.conversationCounter++;
        this.conversationCount.textContent = this.conversationCounter;
    }

    /**
     * Send a predefined quick command
     */
    sendQuickCommand(command) {
        this.userInput.value = command;
        this.sendMessage();
    }

    /**
     * Add message to chat interface
     */
    addMessage(content, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = sender === 'user' ? 'U' : 'H';
        
        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'message-content-wrapper';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Handle HTML content (for images)
        if (content.includes('<img')) {
            messageContent.innerHTML = content;
        } else {
            messageContent.textContent = content;
        }
        
        const metadata = document.createElement('div');
        metadata.className = 'message-metadata';
        metadata.textContent = new Date().toLocaleTimeString();
        
        contentWrapper.appendChild(messageContent);
        contentWrapper.appendChild(metadata);
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentWrapper);
        
        this.chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    /**
     * Show typing indicator
     */
    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message ai typing-indicator';
        typingDiv.id = 'typingIndicator';
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = 'H';
        
        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'message-content-wrapper';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = `
            <div class="typing-animation">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        
        contentWrapper.appendChild(messageContent);
        typingDiv.appendChild(avatar);
        typingDiv.appendChild(contentWrapper);
        
        this.chatMessages.appendChild(typingDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    /**
     * Hide typing indicator
     */
    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
}

// Export for use in main app
window.ChatModule = ChatModule;