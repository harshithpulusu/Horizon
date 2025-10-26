/**
 * Context Manager - Self-contained conversation memory system
 * Uses localStorage for persistence, does NOT modify existing functionality
 */

class ContextManager {
    constructor() {
        this.isEnabled = false;
        this.contextAware = true;
        this.maxConversations = 50;
        this.maxContextLength = 5;
        this.storageKey = 'horizon_conversation_history';
        this.sessionKey = 'horizon_session_id';
        this.settingsKey = 'horizon_context_settings';
        this.apiEndpoint = '/api/multimodal/analyze-context';
        
        // Feature detection
        this.isSupported = this.checkSupport();
        
        if (this.isSupported) {
            this.init();
        }
        
        console.log('🧠 ContextManager:', this.isSupported ? 'Initialized' : 'Not supported');
    }

    checkSupport() {
        /**
         * Check if browser supports required features
         */
        return !!(
            window.localStorage &&
            window.JSON &&
            document.querySelector('.input-container') // Ensure our target elements exist
        );
    }

    init() {
        /**
         * Initialize context management features
         */
        try {
            this.loadSettings();
            this.loadConversationHistory();
            this.createContextIndicator();
            this.restoreSession();
            this.isEnabled = true;
            
            console.log('✅ ContextManager features enabled');
        } catch (error) {
            console.warn('⚠️ ContextManager initialization failed:', error);
            this.isEnabled = false;
        }
    }

    loadSettings() {
        /**
         * Load context settings from localStorage
         */
        try {
            const saved = localStorage.getItem(this.settingsKey);
            if (saved) {
                const settings = JSON.parse(saved);
                this.contextAware = settings.contextAware !== undefined ? settings.contextAware : true;
            }
        } catch (error) {
            console.warn('Failed to load context settings:', error);
            this.contextAware = true;
        }
    }

    saveSettings() {
        /**
         * Save context settings to localStorage
         */
        try {
            const settings = {
                contextAware: this.contextAware,
                lastUpdated: new Date().toISOString()
            };
            localStorage.setItem(this.settingsKey, JSON.stringify(settings));
        } catch (error) {
            console.warn('Failed to save context settings:', error);
        }
    }

    loadConversationHistory() {
        /**
         * Load conversation history from localStorage
         */
        try {
            const saved = localStorage.getItem(this.storageKey);
            this.conversationHistory = saved ? JSON.parse(saved) : [];
            
            // Ensure array and trim if too large
            if (!Array.isArray(this.conversationHistory)) {
                this.conversationHistory = [];
            } else if (this.conversationHistory.length > this.maxConversations) {
                this.conversationHistory = this.conversationHistory.slice(-this.maxConversations);
                this.saveConversationHistory();
            }
            
        } catch (error) {
            console.warn('Failed to load conversation history:', error);
            this.conversationHistory = [];
        }
    }

    saveConversationHistory() {
        /**
         * Save conversation history to localStorage
         */
        try {
            const trimmed = this.conversationHistory.slice(-this.maxConversations);
            localStorage.setItem(this.storageKey, JSON.stringify(trimmed));
        } catch (error) {
            console.warn('Failed to save conversation history:', error);
        }
    }

    createContextIndicator() {
        /**
         * Create context toggle indicator (non-intrusive)
         */
        const inputContainer = document.querySelector('.input-container');
        if (!inputContainer) return;

        const contextIndicator = document.createElement('button');
        contextIndicator.className = 'context-indicator';
        contextIndicator.title = 'Toggle conversation memory (remembers previous conversations)';
        contextIndicator.addEventListener('click', () => this.toggleContextAware());

        // Add to input container near send button
        const sendButton = inputContainer.querySelector('#sendButton');
        if (sendButton) {
            sendButton.parentNode.insertBefore(contextIndicator, sendButton);
        } else {
            inputContainer.appendChild(contextIndicator);
        }

        this.updateContextIndicator();
    }

    updateContextIndicator() {
        /**
         * Update context indicator appearance
         */
        const indicator = document.querySelector('.context-indicator');
        if (!indicator) return;

        if (this.contextAware) {
            indicator.classList.add('context-active');
            indicator.innerHTML = '🧠 ON';
            indicator.title = 'Context awareness enabled - AI remembers previous conversations';
        } else {
            indicator.classList.remove('context-active');
            indicator.innerHTML = '🧠 OFF';
            indicator.title = 'Context awareness disabled - Fresh conversation context';
        }
    }

    toggleContextAware() {
        /**
         * Toggle context awareness on/off
         */
        this.contextAware = !this.contextAware;
        this.updateContextIndicator();
        this.saveSettings();
        
        const message = this.contextAware ? 
            'Context awareness enabled - AI will remember previous conversations' :
            'Context awareness disabled - Fresh conversation context';
        
        this.showNotification(message, 'info');
        
        console.log('🧠 Context awareness:', this.contextAware ? 'enabled' : 'disabled');
    }

    restoreSession() {
        /**
         * Restore session information
         */
        if (this.conversationHistory.length > 0) {
            const lastConversation = this.conversationHistory[this.conversationHistory.length - 1];
            const timeAgo = this.getTimeAgo(new Date(lastConversation.timestamp));
            
            console.log(`🔄 Session restored with ${this.conversationHistory.length} conversations. Last: ${timeAgo}`);
        }
    }

    // CONVERSATION MANAGEMENT

    addConversation(userMessage, aiResponse, metadata = {}) {
        /**
         * Add a conversation to the history
         */
        const conversation = {
            id: Date.now() + Math.random(),
            timestamp: new Date().toISOString(),
            userMessage: userMessage,
            aiResponse: aiResponse,
            sessionId: this.getSessionId(),
            metadata: {
                personality: metadata.personality || 'friendly',
                responseTime: metadata.responseTime || 0,
                hasImages: metadata.hasImages || false,
                imageCount: metadata.imageCount || 0,
                ...metadata
            }
        };

        this.conversationHistory.push(conversation);
        this.saveConversationHistory();
        
        console.log('💬 Conversation added to history');
        return conversation.id;
    }

    getContextForMessage() {
        /**
         * Get conversation context for AI processing
         */
        if (!this.contextAware || this.conversationHistory.length === 0) {
            return {
                hasContext: false,
                contextAware: this.contextAware,
                conversationCount: 0,
                contextPrompt: ''
            };
        }

        // Get recent conversations
        const recentConversations = this.conversationHistory.slice(-this.maxContextLength);
        
        // Build context prompt
        let contextPrompt = '\\n\\n[Previous conversation context]:\\n';
        recentConversations.forEach((conv, index) => {
            const timeAgo = this.getTimeAgo(new Date(conv.timestamp));
            const userPreview = conv.userMessage.substring(0, 100);
            const aiPreview = conv.aiResponse.substring(0, 100);
            
            contextPrompt += `${index + 1}. ${timeAgo}: User: "${userPreview}${userPreview.length < conv.userMessage.length ? '...' : ''}"`;
            if (conv.metadata.hasImages) {
                contextPrompt += ` [with ${conv.metadata.imageCount} image(s)]`;
            }
            contextPrompt += `\\n   AI: "${aiPreview}${aiPreview.length < conv.aiResponse.length ? '...' : ''}"\\n`;
        });
        contextPrompt += '[End of context]\\n\\n';

        return {
            hasContext: true,
            contextAware: this.contextAware,
            conversationCount: this.conversationHistory.length,
            recentCount: recentConversations.length,
            contextPrompt: contextPrompt,
            sessionId: this.getSessionId()
        };
    }

    async analyzeContext() {
        /**
         * Get context analysis from the server
         */
        try {
            const response = await fetch(this.apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    conversation_history: this.conversationHistory.slice(-10), // Last 10 conversations
                    current_message: '',
                    session_id: this.getSessionId()
                })
            });

            const result = await response.json();
            
            if (result.success) {
                return result.context_analysis;
            } else {
                console.warn('Context analysis failed:', result.error);
                return null;
            }
            
        } catch (error) {
            console.warn('Context analysis error:', error);
            return null;
        }
    }

    // SESSION MANAGEMENT

    getSessionId() {
        /**
         * Get or create session ID
         */
        let sessionId = localStorage.getItem(this.sessionKey);
        if (!sessionId) {
            sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem(this.sessionKey, sessionId);
        }
        return sessionId;
    }

    clearHistory() {
        /**
         * Clear conversation history
         */
        this.conversationHistory = [];
        localStorage.removeItem(this.storageKey);
        this.showNotification('Conversation history cleared', 'success');
        console.log('🧹 Conversation history cleared');
    }

    exportHistory() {
        /**
         * Export conversation history as JSON
         */
        const data = {
            exportDate: new Date().toISOString(),
            sessionId: this.getSessionId(),
            conversations: this.conversationHistory,
            totalConversations: this.conversationHistory.length,
            settings: {
                contextAware: this.contextAware,
                maxConversations: this.maxConversations
            }
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `horizon_conversations_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.showNotification('Conversation history exported', 'success');
        console.log('📥 Conversation history exported');
    }

    // UTILITY METHODS

    getTimeAgo(date) {
        /**
         * Get human-readable time difference
         */
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMins / 60);
        const diffDays = Math.floor(diffHours / 24);

        if (diffMins < 1) return 'just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        if (diffDays < 7) return `${diffDays}d ago`;
        return date.toLocaleDateString();
    }

    showNotification(message, type = 'info') {
        /**
         * Show user notification
         */
        const notification = document.createElement('div');
        notification.className = `context-notification context-notification-${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 3000);
    }

    // PUBLIC API METHODS

    enhanceMessage(originalMessage) {
        /**
         * Enhance a message with context if enabled
         */
        if (!this.contextAware) {
            return originalMessage;
        }

        const context = this.getContextForMessage();
        if (context.hasContext) {
            return context.contextPrompt + originalMessage;
        }

        return originalMessage;
    }

    getStatus() {
        /**
         * Get current status for debugging
         */
        return {
            isSupported: this.isSupported,
            isEnabled: this.isEnabled,
            contextAware: this.contextAware,
            conversationCount: this.conversationHistory.length,
            sessionId: this.getSessionId(),
            maxConversations: this.maxConversations
        };
    }
}

// Auto-initialize if DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.ContextManager = new ContextManager();
    });
} else {
    window.ContextManager = new ContextManager();
}