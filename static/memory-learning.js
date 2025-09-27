// Cross-Session Memory Persistence and User Preference Learning
// Advanced AI Memory and Adaptation System

class MemoryLearningSystem {
    constructor(aiAssistant) {
        this.aiAssistant = aiAssistant;
        this.userId = 'default';
        this.currentSessionId = this.generateSessionId();
        this.persistentMemory = new Map();
        this.userPreferences = new Map();
        this.contextBridges = [];
        this.learningEnabled = true;
        
        this.init();
    }
    
    init() {
        this.initializeMemorySystem();
        this.initializePreferenceLearning();
        this.loadPersistentContext();
        this.setupSessionBridges();
        this.addMemoryUI();
        
        console.log('🧠 Memory & Learning System initialized');
    }
    
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    // ===== CROSS-SESSION MEMORY PERSISTENCE =====
    
    initializeMemorySystem() {
        // Set up memory decay and cleanup
        this.memoryDecayInterval = setInterval(() => {
            this.processMemoryDecay();
        }, 300000); // Every 5 minutes
        
        // Set up context relevance scoring
        this.contextScoringEnabled = true;
        
        console.log('🔄 Memory persistence system active');
    }
    
    async storePersistentContext(contextData) {
        // Store context that should persist across sessions
        try {
            const response = await fetch('/api/memory/context/store', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: this.userId,
                    session_id: this.currentSessionId,
                    ...contextData
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                // Update local memory cache
                this.persistentMemory.set(contextData.context_key, {
                    ...contextData,
                    context_id: result.context_id,
                    stored_at: new Date().toISOString()
                });
                
                console.log(`💾 Context stored: ${contextData.context_key}`);
                return result.context_id;
            }
        } catch (error) {
            console.error('Error storing persistent context:', error);
        }
        return null;
    }
    
    async retrievePersistentContext(filters = {}) {
        // Retrieve relevant context for current session
        try {
            const params = new URLSearchParams({
                user_id: this.userId,
                limit: filters.limit || 50,
                ...filters
            });
            
            const response = await fetch(`/api/memory/context/retrieve?${params}`);
            const result = await response.json();
            
            if (result.success) {
                // Update local cache
                result.contexts.forEach(context => {
                    this.persistentMemory.set(context.context_key, context);
                });
                
                console.log(`🔍 Retrieved ${result.count} context entries`);
                return result.contexts;
            }
        } catch (error) {
            console.error('Error retrieving persistent context:', error);
        }
        return [];
    }
    
    async storeConversationMemory(memoryData) {
        // Store important conversation memories
        try {
            const response = await fetch('/api/memory/conversation/store', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: this.userId,
                    session_id: this.currentSessionId,
                    ...memoryData
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                console.log(`🧠 Memory stored: ${memoryData.memory_type}`);
                return result.memory_id;
            }
        } catch (error) {
            console.error('Error storing conversation memory:', error);
        }
        return null;
    }
    
    async createSessionBridge(bridgeData) {
        // Create continuity bridge between sessions
        try {
            const response = await fetch('/api/memory/bridge/create', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: this.userId,
                    current_session_id: this.currentSessionId,
                    ...bridgeData
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.contextBridges.push({
                    bridge_id: result.bridge_id,
                    ...bridgeData,
                    created_at: new Date().toISOString()
                });
                
                console.log(`🌉 Session bridge created: ${bridgeData.bridge_type}`);
                return result.bridge_id;
            }
        } catch (error) {
            console.error('Error creating session bridge:', error);
        }
        return null;
    }
    
    // ===== USER PREFERENCE LEARNING =====
    
    initializePreferenceLearning() {
        this.preferenceLearning = {
            enabled: true,
            learningThreshold: 0.3,
            adaptationRate: 0.1,
            maxPreferences: 200
        };
        
        // Set up behavioral pattern detection
        this.setupBehavioralTracking();
        
        console.log('📊 Preference learning system active');
    }
    
    setupBehavioralTracking() {
        // Track user interaction patterns
        this.interactionPatterns = {
            responseTimePreferences: [],
            topicInterests: new Map(),
            communicationStyle: {},
            timePatterns: new Map(),
            feedbackPatterns: []
        };
        
        // Monitor click patterns, response times, etc.
        this.startBehavioralMonitoring();
    }
    
    startBehavioralMonitoring() {
        // Track user response preferences
        document.addEventListener('click', (event) => {
            this.recordInteraction('click', {
                element: event.target.tagName,
                timestamp: new Date().toISOString(),
                coordinates: { x: event.clientX, y: event.clientY }
            });
        });
        
        // Track typing patterns
        document.addEventListener('keydown', (event) => {
            if (event.target.classList.contains('chat-input')) {
                this.recordInteraction('typing', {
                    key: event.key,
                    timestamp: new Date().toISOString(),
                    inputLength: event.target.value.length
                });
            }
        });
    }
    
    recordInteraction(type, data) {
        // Record interaction for preference learning
        const interaction = {
            type: type,
            data: data,
            timestamp: new Date().toISOString(),
            session_id: this.currentSessionId
        };
        
        // Process for preference patterns
        this.analyzeInteractionForPreferences(interaction);
    }
    
    analyzeInteractionForPreferences(interaction) {
        // Analyze interaction patterns for preference learning
        switch (interaction.type) {
            case 'click':
                this.updateClickPreferences(interaction.data);
                break;
            case 'typing':
                this.updateTypingPreferences(interaction.data);
                break;
            case 'response_time':
                this.updateResponseTimePreferences(interaction.data);
                break;
        }
    }
    
    updateClickPreferences(clickData) {
        // Learn from user click patterns
        const elementType = clickData.element.toLowerCase();
        const currentPrefs = this.interactionPatterns.communicationStyle;
        
        if (!currentPrefs[elementType]) {
            currentPrefs[elementType] = { count: 0, preference_strength: 0.5 };
        }
        
        currentPrefs[elementType].count++;
        currentPrefs[elementType].preference_strength = Math.min(1.0, 
            currentPrefs[elementType].preference_strength + 0.01
        );
        
        // Learn preference if pattern is strong enough
        if (currentPrefs[elementType].count >= 5) {
            this.learnUserPreference({
                preference_category: 'ui_interaction',
                preference_name: `preferred_${elementType}`,
                preference_value: {
                    element_type: elementType,
                    usage_frequency: currentPrefs[elementType].count,
                    preference_strength: currentPrefs[elementType].preference_strength
                },
                learning_source: 'behavioral_analysis',
                confidence_level: Math.min(0.8, currentPrefs[elementType].count / 20)
            });
        }
    }
    
    updateTypingPreferences(typingData) {
        // Learn from typing patterns
        const patterns = this.interactionPatterns;
        
        // Track typing speed and patterns
        const typingSpeed = this.calculateTypingSpeed(typingData);
        
        if (typingSpeed > 0) {
            patterns.responseTimePreferences.push({
                speed: typingSpeed,
                timestamp: typingData.timestamp,
                input_length: typingData.inputLength
            });
            
            // Keep only recent data (last 50 interactions)
            if (patterns.responseTimePreferences.length > 50) {
                patterns.responseTimePreferences.shift();
            }
            
            // Learn typing preference
            const avgSpeed = patterns.responseTimePreferences.reduce((sum, p) => sum + p.speed, 0) / 
                            patterns.responseTimePreferences.length;
            
            this.learnUserPreference({
                preference_category: 'communication',
                preference_name: 'typing_speed_preference',
                preference_value: {
                    average_speed: avgSpeed,
                    preferred_response_pace: avgSpeed > 50 ? 'fast' : avgSpeed > 20 ? 'medium' : 'slow'
                },
                learning_source: 'behavioral_analysis',
                confidence_level: Math.min(0.9, patterns.responseTimePreferences.length / 30)
            });
        }
    }
    
    calculateTypingSpeed(typingData) {
        // Calculate typing speed in characters per minute
        const now = new Date(typingData.timestamp);
        const lastTyping = this.lastTypingTime;
        
        if (lastTyping && typingData.inputLength > this.lastInputLength) {
            const timeDiff = (now - lastTyping) / 1000; // seconds
            const charDiff = typingData.inputLength - this.lastInputLength;
            
            if (timeDiff > 0 && charDiff > 0) {
                this.lastTypingTime = now;
                this.lastInputLength = typingData.inputLength;
                return (charDiff / timeDiff) * 60; // chars per minute
            }
        }
        
        this.lastTypingTime = now;
        this.lastInputLength = typingData.inputLength || 0;
        return 0;
    }
    
    async learnUserPreference(preferenceData) {
        // Learn and store user preferences
        try {
            const response = await fetch('/api/preferences/adaptive/learn', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: this.userId,
                    ...preferenceData
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                // Update local preferences cache
                const prefKey = `${preferenceData.preference_category}.${preferenceData.preference_name}`;
                this.userPreferences.set(prefKey, {
                    ...preferenceData,
                    preference_id: result.preference_id,
                    learned_at: new Date().toISOString()
                });
                
                console.log(`📚 Learned preference: ${prefKey}`);
                
                // Show subtle notification about learning
                this.showLearningNotification(preferenceData.preference_name, result.action);
                
                return result.preference_id;
            }
        } catch (error) {
            console.error('Error learning user preference:', error);
        }
        return null;
    }
    
    async getAdaptivePreferences(category = null, minConfidence = 0.3) {
        // Get user's learned preferences
        try {
            const params = new URLSearchParams({
                user_id: this.userId,
                min_confidence: minConfidence.toString()
            });
            
            if (category) {
                params.append('category', category);
            }
            
            const response = await fetch(`/api/preferences/adaptive/get?${params}`);
            const result = await response.json();
            
            if (result.success) {
                // Update local cache
                result.preferences.forEach(pref => {
                    const prefKey = `${pref.category}.${pref.name}`;
                    this.userPreferences.set(prefKey, pref);
                });
                
                console.log(`📖 Retrieved ${result.count} preferences`);
                return result.preferences;
            }
        } catch (error) {
            console.error('Error getting adaptive preferences:', error);
        }
        return [];
    }
    
    async recordPreferenceFeedback(feedbackData) {
        // Record feedback about AI predictions
        try {
            const response = await fetch('/api/preferences/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: this.userId,
                    session_id: this.currentSessionId,
                    ...feedbackData
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                console.log('✅ Preference feedback recorded');
                return result.feedback_id;
            }
        } catch (error) {
            console.error('Error recording preference feedback:', error);
        }
        return null;
    }
    
    // ===== INTELLIGENT CONTEXT MANAGEMENT =====
    
    async loadPersistentContext() {
        // Load relevant context for current session
        const contexts = await this.retrievePersistentContext({
            limit: 30
        });
        
        // Process and apply context
        contexts.forEach(context => {
            this.applyContextToSession(context);
        });
        
        // Update UI with context indicators
        this.updateContextUI(contexts);
    }
    
    applyContextToSession(context) {
        // Apply persistent context to current session
        switch (context.context_type) {
            case 'personal_info':
                this.applyPersonalContext(context);
                break;
            case 'preferences':
                this.applyPreferenceContext(context);
                break;
            case 'conversation_history':
                this.applyConversationContext(context);
                break;
            case 'topics_discussed':
                this.applyTopicContext(context);
                break;
        }
    }
    
    applyPersonalContext(context) {
        // Apply personal information context
        if (this.aiAssistant && this.aiAssistant.updatePersonalContext) {
            this.aiAssistant.updatePersonalContext(context.context_value);
        }
    }
    
    applyPreferenceContext(context) {
        // Apply preference context
        const prefKey = `${context.context_category}.${context.context_key}`;
        this.userPreferences.set(prefKey, context.context_value);
    }
    
    setupSessionBridges() {
        // Set up bridges for session continuity
        // Check for pending bridges from previous sessions
        this.checkPendingBridges();
        
        // Set up auto-bridge creation for important contexts
        this.setupAutoBridgeCreation();
    }
    
    checkPendingBridges() {
        // Check for context bridges that should be mentioned
        // This would typically query the database for pending bridges
        console.log('🔍 Checking for pending session bridges...');
    }
    
    setupAutoBridgeCreation() {
        // Set up automatic bridge creation for certain conditions
        const conditions = [
            'unresolved_question',
            'follow_up_task',
            'topic_continuation'
        ];
        
        conditions.forEach(condition => {
            this.monitorForBridgeCondition(condition);
        });
    }
    
    monitorForBridgeCondition(condition) {
        // Monitor conversation for bridge-worthy conditions
        if (this.aiAssistant) {
            this.aiAssistant.addEventListener(`detected_${condition}`, (event) => {
                this.createSessionBridge({
                    bridge_type: condition,
                    bridge_data: event.detail,
                    importance_level: this.calculateImportanceLevel(condition, event.detail)
                });
            });
        }
    }
    
    calculateImportanceLevel(condition, data) {
        // Calculate importance level for bridge creation
        const baseImportance = {
            'unresolved_question': 4,
            'follow_up_task': 3,
            'topic_continuation': 2
        };
        
        return baseImportance[condition] || 1;
    }
    
    // ===== MEMORY PROCESSING =====
    
    processMemoryDecay() {
        // Process memory decay and cleanup
        this.persistentMemory.forEach((memory, key) => {
            if (memory.decay_rate && memory.last_referenced) {
                const daysSinceReference = this.daysSince(new Date(memory.last_referenced));
                const decayFactor = Math.pow(1 - memory.decay_rate, daysSinceReference);
                
                if (decayFactor < 0.1) {
                    // Memory has decayed significantly
                    console.log(`🧹 Memory decayed: ${key}`);
                    this.persistentMemory.delete(key);
                } else {
                    // Update memory strength
                    memory.importance_score *= decayFactor;
                }
            }
        });
    }
    
    daysSince(date) {
        const now = new Date();
        const diffTime = Math.abs(now - date);
        return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    }
    
    // ===== UI AND NOTIFICATIONS =====
    
    addMemoryUI() {
        this.addMemoryIndicators();
        this.addPreferenceLearningUI();
        this.addMemoryManagementPanel();
    }
    
    addMemoryIndicators() {
        // Add subtle indicators showing memory and learning status
        const indicator = document.createElement('div');
        indicator.id = 'memoryLearningIndicator';
        indicator.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 20px;
            width: 40px;
            height: 40px;
            background: linear-gradient(45deg, #4ecdc4, #44a08d);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(78, 205, 196, 0.3);
            transition: all 0.3s ease;
        `;
        indicator.innerHTML = '🧠';
        indicator.title = 'Memory & Learning System Active';
        
        indicator.onclick = () => this.toggleMemoryPanel();
        
        document.body.appendChild(indicator);
    }
    
    addPreferenceLearningUI() {
        // Add UI elements for preference learning feedback
        const style = document.createElement('style');
        style.textContent = `
            .learning-notification {
                position: fixed;
                top: 20px;
                right: 20px;
                background: rgba(68, 160, 141, 0.95);
                color: white;
                padding: 12px 18px;
                border-radius: 8px;
                font-size: 0.9em;
                z-index: 10000;
                animation: slideInRight 0.3s ease-out;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            }
            
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            
            .memory-panel {
                position: fixed;
                bottom: 70px;
                left: 20px;
                width: 300px;
                max-height: 400px;
                background: rgba(26, 26, 46, 0.95);
                border: 2px solid #4ecdc4;
                border-radius: 12px;
                padding: 20px;
                color: white;
                font-size: 0.9em;
                z-index: 10000;
                overflow-y: auto;
                backdrop-filter: blur(10px);
                display: none;
            }
            
            .memory-panel h4 {
                color: #4ecdc4;
                margin: 0 0 15px 0;
                text-align: center;
            }
            
            .memory-stat {
                display: flex;
                justify-content: space-between;
                margin: 8px 0;
                padding: 8px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 6px;
            }
            
            .preference-item {
                margin: 10px 0;
                padding: 10px;
                background: rgba(78, 205, 196, 0.1);
                border-radius: 6px;
                border-left: 3px solid #4ecdc4;
            }
        `;
        document.head.appendChild(style);
    }
    
    addMemoryManagementPanel() {
        const panel = document.createElement('div');
        panel.id = 'memoryPanel';
        panel.className = 'memory-panel';
        panel.innerHTML = `
            <h4>🧠 Memory & Learning</h4>
            <div id="memoryStats">
                <div class="memory-stat">
                    <span>Persistent Contexts:</span>
                    <span id="contextCount">0</span>
                </div>
                <div class="memory-stat">
                    <span>Learned Preferences:</span>
                    <span id="preferenceCount">0</span>
                </div>
                <div class="memory-stat">
                    <span>Session Bridges:</span>
                    <span id="bridgeCount">0</span>
                </div>
                <div class="memory-stat">
                    <span>Learning Status:</span>
                    <span id="learningStatus">Active</span>
                </div>
            </div>
            <div id="recentPreferences">
                <h5 style="color: #4ecdc4; margin: 15px 0 10px 0;">Recent Learning:</h5>
                <div id="preferencesList"></div>
            </div>
            <div style="text-align: center; margin-top: 15px;">
                <button onclick="window.memoryLearning.clearMemory()" 
                        style="background: #ff6b6b; border: none; color: white; padding: 6px 12px; border-radius: 4px; cursor: pointer;">
                    Clear Memory
                </button>
            </div>
        `;
        
        document.body.appendChild(panel);
        
        // Update stats periodically
        this.updateMemoryStats();
        setInterval(() => this.updateMemoryStats(), 10000);
    }
    
    toggleMemoryPanel() {
        const panel = document.getElementById('memoryPanel');
        if (panel) {
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
        }
    }
    
    updateMemoryStats() {
        document.getElementById('contextCount').textContent = this.persistentMemory.size;
        document.getElementById('preferenceCount').textContent = this.userPreferences.size;
        document.getElementById('bridgeCount').textContent = this.contextBridges.length;
        document.getElementById('learningStatus').textContent = this.learningEnabled ? 'Active' : 'Paused';
        
        this.updateRecentPreferences();
    }
    
    updateRecentPreferences() {
        const list = document.getElementById('preferencesList');
        if (!list) return;
        
        list.innerHTML = '';
        
        // Show recent preferences
        const recentPrefs = Array.from(this.userPreferences.entries())
            .slice(-5)
            .reverse();
        
        recentPrefs.forEach(([key, pref]) => {
            const item = document.createElement('div');
            item.className = 'preference-item';
            item.innerHTML = `
                <strong>${pref.name || key}</strong><br>
                <small>Confidence: ${Math.round((pref.confidence_level || 0.5) * 100)}%</small>
            `;
            list.appendChild(item);
        });
    }
    
    updateContextUI(contexts) {
        // Update UI to show active context
        const activeContexts = contexts.filter(c => c.importance_score > 0.7);
        
        if (activeContexts.length > 0 && this.aiAssistant) {
            // Show context indicator in chat
            this.showContextIndicator(activeContexts);
        }
    }
    
    showContextIndicator(contexts) {
        // Show subtle indicator that context is being used
        const indicator = document.createElement('div');
        indicator.className = 'context-indicator';
        indicator.innerHTML = `💭 Using ${contexts.length} context${contexts.length > 1 ? 's' : ''} from previous sessions`;
        indicator.style.cssText = `
            background: rgba(78, 205, 196, 0.1);
            border: 1px solid rgba(78, 205, 196, 0.3);
            border-radius: 6px;
            padding: 8px 12px;
            margin: 10px 0;
            font-size: 0.8em;
            color: #4ecdc4;
            text-align: center;
        `;
        
        // Add to chat container
        const chatContainer = document.querySelector('.chat-container') || document.querySelector('.messages');
        if (chatContainer) {
            chatContainer.insertBefore(indicator, chatContainer.firstChild);
            
            // Remove after 5 seconds
            setTimeout(() => {
                if (indicator.parentNode) {
                    indicator.parentNode.removeChild(indicator);
                }
            }, 5000);
        }
    }
    
    showLearningNotification(preferenceName, action) {
        const notification = document.createElement('div');
        notification.className = 'learning-notification';
        notification.innerHTML = `🧠 ${action === 'created' ? 'Learned' : 'Updated'}: ${preferenceName.replace('_', ' ')}`;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 3000);
    }
    
    // ===== PUBLIC API =====
    
    clearMemory() {
        if (confirm('Clear all learned preferences and memory? This cannot be undone.')) {
            this.persistentMemory.clear();
            this.userPreferences.clear();
            this.contextBridges = [];
            this.updateMemoryStats();
            console.log('🧹 Memory cleared');
        }
    }
    
    toggleLearning() {
        this.learningEnabled = !this.learningEnabled;
        document.getElementById('learningStatus').textContent = this.learningEnabled ? 'Active' : 'Paused';
        console.log(`📚 Learning ${this.learningEnabled ? 'enabled' : 'disabled'}`);
    }
    
    getMemoryStats() {
        return {
            persistent_contexts: this.persistentMemory.size,
            learned_preferences: this.userPreferences.size,
            session_bridges: this.contextBridges.length,
            learning_enabled: this.learningEnabled,
            current_session: this.currentSessionId
        };
    }
    
    // Integration with main AI assistant
    onConversationEnd(conversationData) {
        // Store important conversation elements
        if (conversationData.important_facts) {
            this.storeConversationMemory({
                memory_type: 'fact',
                memory_content: conversationData.important_facts,
                memory_summary: conversationData.summary,
                relevance_score: 0.8
            });
        }
        
        // Create bridges for unresolved items
        if (conversationData.unresolved_questions) {
            this.createSessionBridge({
                bridge_type: 'unresolved_question',
                bridge_data: conversationData.unresolved_questions,
                importance_level: 4
            });
        }
    }
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        if (window.aiAssistant) {
            window.memoryLearning = new MemoryLearningSystem(window.aiAssistant);
            console.log('🚀 Memory & Learning system loaded successfully!');
        }
    }, 1500);
});