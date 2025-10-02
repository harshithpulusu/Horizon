/**
 * Predictive Assistance Module for Horizon AI
 * Handles proactive suggestions, pattern learning, and user need anticipation
 */

class PredictiveAssistance {
    constructor() {
        this.userId = null;
        this.sessionId = null;
        this.suggestions = [];
        this.patterns = [];
        this.feedbackQueue = [];
        this.isEnabled = true;
        this.lastSuggestionTime = 0;
        this.suggestionCooldown = 30000; // 30 seconds
        
        this.init();
    }
    
    init() {
        console.log('🔮 Initializing Predictive Assistance');
        this.setupEventListeners();
        this.loadUserPreferences();
        this.checkPredictiveStatus();
    }
    
    setupEventListeners() {
        // Listen for chat responses to analyze patterns
        document.addEventListener('chatResponse', (event) => {
            this.analyzeInteraction(event.detail);
        });
        
        // Listen for user context changes
        document.addEventListener('contextChange', (event) => {
            this.updateContext(event.detail);
        });
        
        // Periodic suggestion checks
        setInterval(() => this.checkForSuggestions(), 60000); // Every minute
    }
    
    async checkPredictiveStatus() {
        try {
            const response = await fetch('/api/predictive/status');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.isEnabled = data.predictive_status.predictive_assistance_available;
                console.log(`🔮 Predictive assistance: ${this.isEnabled ? 'enabled' : 'disabled'}`);
                
                if (this.isEnabled) {
                    this.displayPredictiveStatus(data.predictive_status);
                }
            }
        } catch (error) {
            console.error('Failed to check predictive status:', error);
            this.isEnabled = false;
        }
    }
    
    setUser(userId, sessionId) {
        this.userId = userId;
        this.sessionId = sessionId;
        
        if (this.isEnabled) {
            // Start pattern analysis for this user
            this.analyzeUserBehavior();
        }
    }
    
    async analyzeUserBehavior() {
        if (!this.userId || !this.isEnabled) return;
        
        try {
            const response = await fetch('/api/predictive/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: this.userId,
                    timeframe_days: 30
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                console.log(`📊 Analyzed patterns: ${data.analysis_result.patterns_found} found`);
                this.patterns = data.analysis_result.patterns || [];
                
                // Get initial suggestions
                await this.getSuggestions();
            }
        } catch (error) {
            console.error('Behavior analysis failed:', error);
        }
    }
    
    async getSuggestions(context = {}) {
        if (!this.userId || !this.isEnabled) return;
        
        // Respect cooldown period
        const now = Date.now();
        if (now - this.lastSuggestionTime < this.suggestionCooldown) {
            return;
        }
        
        try {
            const currentContext = {
                timestamp: new Date().toISOString(),
                hour: new Date().getHours(),
                day_of_week: new Date().getDay(),
                session_id: this.sessionId,
                ...context
            };
            
            const response = await fetch('/api/predictive/suggestions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: this.userId,
                    context: currentContext
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.suggestions = data.suggestions.proactive_suggestions || [];
                this.lastSuggestionTime = now;
                
                if (this.suggestions.length > 0) {
                    this.displaySuggestions();
                }
            }
        } catch (error) {
            console.error('Failed to get suggestions:', error);
        }
    }
    
    displaySuggestions() {
        if (!this.suggestions.length) return;
        
        // Remove existing suggestion panel
        const existingPanel = document.querySelector('.predictive-suggestions');
        if (existingPanel) {
            existingPanel.remove();
        }
        
        // Create suggestion panel
        const panel = document.createElement('div');
        panel.className = 'predictive-suggestions';
        panel.innerHTML = `
            <div class="suggestion-header">
                <span class="suggestion-icon">🔮</span>
                <span class="suggestion-title">Suggested for you</span>
                <button class="suggestion-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
            <div class="suggestion-list">
                ${this.suggestions.map((suggestion, index) => `
                    <div class="suggestion-item" data-index="${index}">
                        <div class="suggestion-content">
                            <div class="suggestion-text">${suggestion.suggestion}</div>
                            <div class="suggestion-meta">
                                <span class="suggestion-confidence">Confidence: ${(suggestion.confidence * 100).toFixed(0)}%</span>
                                <span class="suggestion-urgency urgency-${suggestion.urgency}">${suggestion.urgency}</span>
                            </div>
                        </div>
                        <div class="suggestion-actions">
                            <button class="suggestion-accept" onclick="predictiveAssistance.acceptSuggestion(${index})">
                                ✓ Accept
                            </button>
                            <button class="suggestion-dismiss" onclick="predictiveAssistance.dismissSuggestion(${index})">
                                ✕ Dismiss
                            </button>
                        </div>
                    </div>
                `).join('')}
            </div>
            <div class="suggestion-footer">
                <small>Based on your usage patterns • <a href="#" onclick="predictiveAssistance.showPatternsModal()">View patterns</a></small>
            </div>
        `;
        
        // Add to page
        const chatContainer = document.querySelector('.chat-container') || document.body;
        chatContainer.appendChild(panel);
        
        // Animate in
        requestAnimationFrame(() => {
            panel.classList.add('visible');
        });
        
        // Auto-hide after 30 seconds if no interaction
        setTimeout(() => {
            if (panel.parentElement) {
                panel.classList.add('fading');
                setTimeout(() => panel.remove(), 300);
            }
        }, 30000);
    }
    
    acceptSuggestion(index) {
        const suggestion = this.suggestions[index];
        if (!suggestion) return;
        
        // Execute the suggestion (could trigger a chat message, action, etc.)
        this.executeSuggestion(suggestion);
        
        // Provide positive feedback
        this.provideFeedback(suggestion.type, true, 'User accepted suggestion');
        
        // Remove suggestion panel
        document.querySelector('.predictive-suggestions')?.remove();
        
        console.log('✅ Suggestion accepted:', suggestion.suggestion);
    }
    
    dismissSuggestion(index) {
        const suggestion = this.suggestions[index];
        if (!suggestion) return;
        
        // Provide negative feedback
        this.provideFeedback(suggestion.type, false, 'User dismissed suggestion');
        
        // Remove suggestion item
        const item = document.querySelector(`[data-index="${index}"]`);
        if (item) {
            item.style.opacity = '0.5';
            item.style.pointerEvents = 'none';
        }
        
        console.log('❌ Suggestion dismissed:', suggestion.suggestion);
    }
    
    async provideFeedback(predictionType, wasHelpful, feedback = '') {
        if (!this.userId || !this.isEnabled) return;
        
        try {
            const response = await fetch('/api/predictive/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: this.userId,
                    prediction_type: predictionType,
                    was_helpful: wasHelpful,
                    feedback: feedback
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                console.log('📝 Feedback provided successfully');
            }
        } catch (error) {
            console.error('Failed to provide feedback:', error);
        }
    }
    
    executeSuggestion(suggestion) {
        // Execute based on suggestion type
        switch (suggestion.type) {
            case 'temporal_interaction':
                this.showGreeting();
                break;
            case 'topic_assistance':
                this.showTopicHelp(suggestion);
                break;
            case 'style_adaptation':
                this.adaptInterface(suggestion);
                break;
            case 'work_context':
                this.showWorkTools();
                break;
            case 'weekend_context':
                this.showLeisureOptions();
                break;
            default:
                this.triggerChatMessage(suggestion.suggestion);
        }
    }
    
    showGreeting() {
        const greetings = [
            "Hello! I'm here to help. What can I assist you with today?",
            "Hi there! Ready to tackle your day together?",
            "Good to see you! What's on your mind?",
            "Welcome back! How can I help you today?"
        ];
        
        const greeting = greetings[Math.floor(Math.random() * greetings.length)];
        this.addSystemMessage(greeting);
    }
    
    showTopicHelp(suggestion) {
        this.addSystemMessage(`I notice you often ask about this topic. ${suggestion.suggestion}`);
    }
    
    adaptInterface(suggestion) {
        // Adapt UI based on preferred interaction style
        const style = suggestion.context?.preferred_style;
        if (style) {
            document.body.setAttribute('data-preferred-style', style);
            this.addSystemMessage(`I've adapted the interface to your ${style} style.`);
        }
    }
    
    showWorkTools() {
        this.addSystemMessage("I see you're in work mode! I can help with tasks, scheduling, writing, or quick research. What would you like to focus on?");
    }
    
    showLeisureOptions() {
        this.addSystemMessage("Weekend vibes! I can suggest entertainment, help with personal projects, or just chat. What sounds good?");
    }
    
    triggerChatMessage(message) {
        // Add the suggestion as a system message
        this.addSystemMessage(message);
    }
    
    addSystemMessage(message) {
        // Add a system message to the chat
        const event = new CustomEvent('addSystemMessage', {
            detail: { message, type: 'predictive' }
        });
        document.dispatchEvent(event);
    }
    
    analyzeInteraction(detail) {
        // Analyze user interaction patterns
        const context = {
            input: detail.input,
            response: detail.response,
            personality: detail.personality,
            timestamp: new Date().toISOString(),
            confidence: detail.confidence,
            intent: detail.intent
        };
        
        // Update context for future suggestions
        this.updateContext(context);
        
        // Check if we should get new suggestions based on this interaction
        if (this.shouldGetNewSuggestions(context)) {
            setTimeout(() => this.getSuggestions(context), 2000); // Wait 2 seconds
        }
    }
    
    shouldGetNewSuggestions(context) {
        // Get new suggestions if:
        // 1. User asked about something new
        // 2. Conversation has been going for a while
        // 3. User seems stuck or confused
        
        const confusedKeywords = ['help', 'don\'t know', 'confused', 'stuck', 'what should'];
        const input = context.input?.toLowerCase() || '';
        
        return confusedKeywords.some(keyword => input.includes(keyword)) ||
               context.confidence < 0.6 ||
               Math.random() < 0.1; // 10% chance for serendipitous suggestions
    }
    
    updateContext(context) {
        // Store context for pattern analysis
        this.currentContext = {
            ...this.currentContext,
            ...context,
            timestamp: new Date().toISOString()
        };
    }
    
    checkForSuggestions() {
        // Periodic check for new suggestions based on time and context
        const now = new Date();
        const hour = now.getHours();
        
        // Check for time-based suggestions
        if (hour === 9 && now.getMinutes() === 0) {
            this.getSuggestions({ trigger: 'morning_routine' });
        } else if (hour === 13 && now.getMinutes() === 0) {
            this.getSuggestions({ trigger: 'lunch_break' });
        } else if (hour === 17 && now.getMinutes() === 0) {
            this.getSuggestions({ trigger: 'end_of_workday' });
        }
    }
    
    async showPatternsModal() {
        if (!this.userId || !this.isEnabled) return;
        
        try {
            const response = await fetch(`/api/predictive/patterns?user_id=${encodeURIComponent(this.userId)}`);
            const data = await response.json();
            
            if (data.status === 'success') {
                this.displayPatternsModal(data.patterns);
            }
        } catch (error) {
            console.error('Failed to load patterns:', error);
        }
    }
    
    displayPatternsModal(patterns) {
        const modal = document.createElement('div');
        modal.className = 'patterns-modal';
        modal.innerHTML = `
            <div class="modal-overlay" onclick="this.parentElement.remove()"></div>
            <div class="modal-content">
                <div class="modal-header">
                    <h3>🧠 Your Behavior Patterns</h3>
                    <button class="modal-close" onclick="this.closest('.patterns-modal').remove()">×</button>
                </div>
                <div class="modal-body">
                    ${patterns.length === 0 ? 
                        '<p>No patterns discovered yet. Keep using Horizon AI to build your profile!</p>' :
                        patterns.map(pattern => `
                            <div class="pattern-item">
                                <h4>${this.formatPatternType(pattern.pattern_type)}</h4>
                                <div class="pattern-details">
                                    <span class="pattern-frequency">Frequency: ${(pattern.frequency * 100).toFixed(0)}%</span>
                                    <span class="pattern-success">Success: ${(pattern.success_rate * 100).toFixed(0)}%</span>
                                </div>
                                <p class="pattern-description">${this.getPatternDescription(pattern)}</p>
                            </div>
                        `).join('')
                    }
                </div>
                <div class="modal-footer">
                    <p><small>These patterns help me provide better suggestions. Data is private and only used to improve your experience.</small></p>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }
    
    formatPatternType(type) {
        return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
    
    getPatternDescription(pattern) {
        switch (pattern.pattern_type) {
            case 'temporal':
                return `You typically interact at ${pattern.pattern_data.typical_time}`;
            case 'topic':
                return `You frequently discuss ${pattern.pattern_data.context_triggers.join(', ')}`;
            case 'interaction_style':
                return `You prefer ${pattern.pattern_data.context_triggers[0]} interactions`;
            default:
                return 'Behavioral pattern detected in your usage';
        }
    }
    
    displayPredictiveStatus(status) {
        // Add a subtle indicator that predictive assistance is active
        const indicator = document.createElement('div');
        indicator.className = 'predictive-indicator';
        indicator.innerHTML = `
            <span class="indicator-icon">🔮</span>
            <span class="indicator-text">Predictive AI Active</span>
        `;
        indicator.title = 'Horizon AI is learning your patterns to provide better assistance';
        
        const header = document.querySelector('.header') || document.querySelector('.container');
        if (header) {
            header.appendChild(indicator);
        }
    }
    
    loadUserPreferences() {
        // Load user preferences for predictive assistance
        const prefs = localStorage.getItem('predictive_preferences');
        if (prefs) {
            try {
                const preferences = JSON.parse(prefs);
                this.isEnabled = preferences.enabled !== false;
                this.suggestionCooldown = preferences.cooldown || 30000;
            } catch (error) {
                console.error('Failed to load preferences:', error);
            }
        }
    }
    
    saveUserPreferences() {
        const preferences = {
            enabled: this.isEnabled,
            cooldown: this.suggestionCooldown
        };
        localStorage.setItem('predictive_preferences', JSON.stringify(preferences));
    }
    
    toggle() {
        this.isEnabled = !this.isEnabled;
        this.saveUserPreferences();
        
        const indicator = document.querySelector('.predictive-indicator');
        if (indicator) {
            indicator.style.opacity = this.isEnabled ? '1' : '0.5';
        }
        
        console.log(`🔮 Predictive assistance ${this.isEnabled ? 'enabled' : 'disabled'}`);
    }
}

// CSS styles for predictive assistance UI
const predictiveStyles = `
    .predictive-suggestions {
        position: fixed;
        top: 20px;
        right: 20px;
        width: 350px;
        background: rgba(0, 0, 0, 0.95);
        border: 1px solid rgba(120, 119, 198, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        z-index: 1000;
        opacity: 0;
        transform: translateX(100px);
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    .predictive-suggestions.visible {
        opacity: 1;
        transform: translateX(0);
    }
    
    .predictive-suggestions.fading {
        opacity: 0;
        transform: translateX(50px);
    }
    
    .suggestion-header {
        display: flex;
        align-items: center;
        padding: 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .suggestion-icon {
        font-size: 18px;
        margin-right: 8px;
    }
    
    .suggestion-title {
        flex: 1;
        font-weight: 600;
        color: #ffffff;
    }
    
    .suggestion-close {
        background: none;
        border: none;
        color: #888;
        cursor: pointer;
        font-size: 20px;
        padding: 0;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .suggestion-close:hover {
        color: #ffffff;
    }
    
    .suggestion-item {
        padding: 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .suggestion-item:last-child {
        border-bottom: none;
    }
    
    .suggestion-content {
        margin-bottom: 10px;
    }
    
    .suggestion-text {
        color: #ffffff;
        margin-bottom: 8px;
        line-height: 1.4;
    }
    
    .suggestion-meta {
        display: flex;
        gap: 10px;
        font-size: 12px;
    }
    
    .suggestion-confidence {
        color: #7777c6;
    }
    
    .suggestion-urgency {
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .urgency-low { background: rgba(76, 175, 80, 0.2); color: #4caf50; }
    .urgency-medium { background: rgba(255, 193, 7, 0.2); color: #ffc107; }
    .urgency-high { background: rgba(255, 87, 34, 0.2); color: #ff5722; }
    .urgency-urgent { background: rgba(244, 67, 54, 0.2); color: #f44336; }
    
    .suggestion-actions {
        display: flex;
        gap: 8px;
    }
    
    .suggestion-accept, .suggestion-dismiss {
        padding: 6px 12px;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-size: 12px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .suggestion-accept {
        background: #4caf50;
        color: white;
    }
    
    .suggestion-accept:hover {
        background: #45a049;
    }
    
    .suggestion-dismiss {
        background: rgba(255, 255, 255, 0.1);
        color: #ccc;
    }
    
    .suggestion-dismiss:hover {
        background: rgba(255, 255, 255, 0.2);
        color: #fff;
    }
    
    .suggestion-footer {
        padding: 10px 15px;
        color: #888;
        font-size: 11px;
    }
    
    .suggestion-footer a {
        color: #7777c6;
        text-decoration: none;
    }
    
    .suggestion-footer a:hover {
        text-decoration: underline;
    }
    
    .predictive-indicator {
        position: fixed;
        top: 10px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(119, 119, 198, 0.2);
        border: 1px solid rgba(119, 119, 198, 0.3);
        border-radius: 20px;
        padding: 5px 12px;
        font-size: 12px;
        color: #7777c6;
        z-index: 999;
        backdrop-filter: blur(5px);
    }
    
    .indicator-icon {
        margin-right: 5px;
    }
    
    .patterns-modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: 2000;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .modal-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(5px);
    }
    
    .modal-content {
        position: relative;
        background: #1a1a2e;
        border: 1px solid rgba(120, 119, 198, 0.3);
        border-radius: 12px;
        width: 90%;
        max-width: 600px;
        max-height: 80vh;
        overflow-y: auto;
    }
    
    .modal-header {
        display: flex;
        align-items: center;
        padding: 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .modal-header h3 {
        flex: 1;
        margin: 0;
        color: #ffffff;
    }
    
    .modal-close {
        background: none;
        border: none;
        color: #888;
        cursor: pointer;
        font-size: 24px;
        padding: 0;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .modal-body {
        padding: 20px;
    }
    
    .pattern-item {
        margin-bottom: 20px;
        padding: 15px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }
    
    .pattern-item h4 {
        margin: 0 0 8px 0;
        color: #7777c6;
    }
    
    .pattern-details {
        display: flex;
        gap: 15px;
        margin-bottom: 8px;
        font-size: 12px;
    }
    
    .pattern-frequency, .pattern-success {
        color: #ccc;
    }
    
    .pattern-description {
        margin: 0;
        color: #ffffff;
        font-size: 14px;
    }
    
    .modal-footer {
        padding: 15px 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: #888;
    }
    
    @media (max-width: 768px) {
        .predictive-suggestions {
            width: calc(100vw - 40px);
            right: 20px;
            left: 20px;
        }
        
        .modal-content {
            width: 95%;
            margin: 20px;
        }
    }
`;

// Inject styles
const styleSheet = document.createElement('style');
styleSheet.textContent = predictiveStyles;
document.head.appendChild(styleSheet);

// Global instance
window.predictiveAssistance = new PredictiveAssistance();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PredictiveAssistance;
}